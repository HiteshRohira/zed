use anyhow::{Context as _, Result, anyhow, bail};
use base64::Engine as _;
use credentials_provider::CredentialsProvider;
use futures::AsyncReadExt as _;
use futures::channel::oneshot;
use gpui::AsyncApp;
use http_client::{AsyncBody, HttpClient, Request};
use rand::Rng as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tiny_http::Header;
use url::Url;
use util::ResultExt as _;

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const JWT_AUTH_CLAIM_PATH: &str = "https://api.openai.com/auth";
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(2 * 60);
const REFRESH_SKEW: Duration = Duration::from_secs(60);
pub const CREDENTIALS_KEY: &str = "zed://openai/codex-oauth";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodexAuthSession {
    pub access_token: String,
    pub refresh_token: String,
    pub account_id: String,
    pub expires_at_unix_ms: u64,
}

impl CodexAuthSession {
    pub fn should_refresh(&self) -> bool {
        let refresh_at = self
            .expires_at()
            .unwrap_or(UNIX_EPOCH)
            .checked_sub(REFRESH_SKEW)
            .unwrap_or(UNIX_EPOCH);
        SystemTime::now() >= refresh_at
    }

    fn expires_at(&self) -> Option<SystemTime> {
        UNIX_EPOCH.checked_add(Duration::from_millis(self.expires_at_unix_ms))
    }
}

#[derive(Debug)]
pub struct PendingCodexOAuthFlow {
    pub authorization_url: String,
    state: String,
    pkce_verifier: String,
    callback_rx: oneshot::Receiver<Result<OAuthCallback>>,
}

impl PendingCodexOAuthFlow {
    pub async fn finish(self, http_client: Arc<dyn HttpClient>) -> Result<CodexAuthSession> {
        let callback = self
            .callback_rx
            .await
            .map_err(|_| anyhow!("OAuth callback server stopped before receiving a response"))?
            .context("OAuth callback server received an invalid request")?;

        if callback.state != self.state {
            bail!("OAuth state parameter mismatch");
        }

        exchange_authorization_code(
            http_client,
            &callback.code,
            &self.pkce_verifier,
            REDIRECT_URI,
        )
        .await
    }
}

#[derive(Debug)]
struct OAuthCallback {
    code: String,
    state: String,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    expires_in: u64,
}

#[derive(Debug)]
struct PkceChallenge {
    verifier: String,
    challenge: String,
}

pub fn begin_oauth_flow() -> Result<PendingCodexOAuthFlow> {
    let pkce = generate_pkce_challenge();
    let state = random_state();
    let authorization_url = build_authorization_url(&pkce, &state)?;
    let callback_rx = start_callback_server()?;

    Ok(PendingCodexOAuthFlow {
        authorization_url,
        state,
        pkce_verifier: pkce.verifier,
        callback_rx,
    })
}

pub async fn load_session(
    credentials_provider: &dyn CredentialsProvider,
    cx: &AsyncApp,
) -> Result<Option<CodexAuthSession>> {
    let Some((_, bytes)) = credentials_provider
        .read_credentials(CREDENTIALS_KEY, cx)
        .await?
    else {
        return Ok(None);
    };

    let session: CodexAuthSession =
        serde_json::from_slice(&bytes).context("failed to parse stored Codex auth session")?;
    Ok(Some(session))
}

pub async fn store_session(
    credentials_provider: &dyn CredentialsProvider,
    session: &CodexAuthSession,
    cx: &AsyncApp,
) -> Result<()> {
    let bytes = serde_json::to_vec(session)?;
    credentials_provider
        .write_credentials(CREDENTIALS_KEY, "Bearer", &bytes, cx)
        .await
}

pub async fn delete_session(
    credentials_provider: &dyn CredentialsProvider,
    cx: &AsyncApp,
) -> Result<()> {
    credentials_provider
        .delete_credentials(CREDENTIALS_KEY, cx)
        .await
}

pub async fn refresh_session(
    http_client: Arc<dyn HttpClient>,
    session: &CodexAuthSession,
) -> Result<CodexAuthSession> {
    let params = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "refresh_token")
        .append_pair("refresh_token", &session.refresh_token)
        .append_pair("client_id", CLIENT_ID)
        .finish();

    let response = send_token_request(http_client, params).await?;
    session_from_token_response(response)
}

fn random_state() -> String {
    let mut bytes = [0u8; 16];
    rand::rng().fill(&mut bytes);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn generate_pkce_challenge() -> PkceChallenge {
    let mut random_bytes = [0u8; 32];
    rand::rng().fill(&mut random_bytes);
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let verifier = engine.encode(random_bytes);
    let digest = Sha256::digest(verifier.as_bytes());
    let challenge = engine.encode(digest);
    PkceChallenge {
        verifier,
        challenge,
    }
}

fn build_authorization_url(pkce: &PkceChallenge, state: &str) -> Result<String> {
    let mut url = Url::parse(AUTHORIZE_URL)?;
    {
        let mut query = url.query_pairs_mut();
        query.append_pair("response_type", "code");
        query.append_pair("client_id", CLIENT_ID);
        query.append_pair("redirect_uri", REDIRECT_URI);
        query.append_pair("scope", "openid profile email offline_access");
        query.append_pair("code_challenge", &pkce.challenge);
        query.append_pair("code_challenge_method", "S256");
        query.append_pair("state", state);
        query.append_pair("id_token_add_organizations", "true");
        query.append_pair("codex_cli_simplified_flow", "true");
        query.append_pair("originator", "codex_cli_rs");
    }
    Ok(url.to_string())
}

fn start_callback_server() -> Result<oneshot::Receiver<Result<OAuthCallback>>> {
    let server = tiny_http::Server::http("127.0.0.1:1455").map_err(|error| {
        anyhow!(error).context("failed to bind 127.0.0.1:1455 for OAuth callback")
    })?;
    let (tx, rx) = oneshot::channel();

    std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + CALLBACK_TIMEOUT;

        loop {
            if tx.is_canceled() {
                return;
            }

            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                let _ = tx.send(Err(anyhow!("Timed out waiting for ChatGPT login")));
                return;
            }

            let timeout = remaining.min(Duration::from_millis(500));
            let Some(request) = (match server.recv_timeout(timeout) {
                Ok(request) => request,
                Err(error) => {
                    let _ = tx.send(Err(
                        anyhow!(error).context("OAuth callback server I/O error")
                    ));
                    return;
                }
            }) else {
                continue;
            };

            let result = handle_callback_request(&request);
            let (status_code, body) = match &result {
                Ok(_) => (
                    200,
                    "<html><body><h1>Authorization successful</h1><p>You can close this tab and return to Zed.</p></body></html>",
                ),
                Err(_) => (
                    400,
                    "<html><body><h1>Authorization failed</h1><p>Return to Zed and try again.</p></body></html>",
                ),
            };

            let response = tiny_http::Response::from_string(body)
                .with_status_code(status_code)
                .with_header(
                    Header::from_str("Content-Type: text/html")
                        .expect("failed to construct Content-Type header"),
                )
                .with_header(
                    Header::from_str("Keep-Alive: timeout=0,max=0")
                        .expect("failed to construct Keep-Alive header"),
                );
            request.respond(response).log_err();

            let _ = tx.send(result);
            return;
        }
    });

    Ok(rx)
}

fn handle_callback_request(request: &tiny_http::Request) -> Result<OAuthCallback> {
    let url = Url::parse(&format!("http://localhost{}", request.url()))
        .context("malformed callback request URL")?;
    if url.path() != "/auth/callback" {
        bail!("unexpected OAuth callback path: {}", url.path());
    }

    let query = url
        .query()
        .ok_or_else(|| anyhow!("OAuth callback is missing query parameters"))?;
    let mut code = None;
    let mut state = None;
    let mut error = None;
    let mut error_description = None;

    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        match key.as_ref() {
            "code" if !value.is_empty() => code = Some(value.into_owned()),
            "state" if !value.is_empty() => state = Some(value.into_owned()),
            "error" if !value.is_empty() => error = Some(value.into_owned()),
            "error_description" if !value.is_empty() => {
                error_description = Some(value.into_owned())
            }
            _ => {}
        }
    }

    if let Some(error) = error {
        bail!(
            "OAuth authorization failed: {} ({})",
            error,
            error_description.as_deref().unwrap_or("no description")
        );
    }

    Ok(OAuthCallback {
        code: code.ok_or_else(|| anyhow!("OAuth callback missing code"))?,
        state: state.ok_or_else(|| anyhow!("OAuth callback missing state"))?,
    })
}

async fn exchange_authorization_code(
    http_client: Arc<dyn HttpClient>,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<CodexAuthSession> {
    let params = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "authorization_code")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("code", code)
        .append_pair("code_verifier", verifier)
        .append_pair("redirect_uri", redirect_uri)
        .finish();

    let response = send_token_request(http_client, params).await?;
    session_from_token_response(response)
}

async fn send_token_request(
    http_client: Arc<dyn HttpClient>,
    params: String,
) -> Result<TokenResponse> {
    let request = Request::builder()
        .method(http_client::http::Method::POST)
        .uri(TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .body(AsyncBody::from(params.into_bytes()))?;

    let mut response = http_client.send(request).await?;
    let mut body = String::new();
    response.body_mut().read_to_string(&mut body).await?;

    if !response.status().is_success() {
        bail!(
            "Codex OAuth token request failed with status {}: {}",
            response.status(),
            body
        );
    }

    serde_json::from_str(&body).context("failed to parse Codex OAuth token response")
}

fn session_from_token_response(response: TokenResponse) -> Result<CodexAuthSession> {
    let account_id = account_id_from_access_token(&response.access_token)
        .context("missing ChatGPT account id")?;
    let expires_at = SystemTime::now()
        .checked_add(Duration::from_secs(response.expires_in))
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis() as u64)
        .ok_or_else(|| anyhow!("failed to compute token expiration"))?;

    Ok(CodexAuthSession {
        access_token: response.access_token,
        refresh_token: response.refresh_token,
        account_id,
        expires_at_unix_ms: expires_at,
    })
}

fn account_id_from_access_token(access_token: &str) -> Option<String> {
    let payload = access_token.split('.').nth(1)?;
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let bytes = engine
        .decode(payload)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(payload))
        .ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    value
        .get(JWT_AUTH_CLAIM_PATH)?
        .get("chatgpt_account_id")?
        .as_str()
        .map(ToOwned::to_owned)
}
