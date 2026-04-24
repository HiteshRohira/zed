mod codex;

use anyhow::Result;
use collections::BTreeMap;
use credentials_provider::CredentialsProvider;
use futures::{FutureExt, StreamExt, future::BoxFuture};
use gpui::{AnyView, App, AsyncApp, Context, Entity, SharedString, Task, Window};
use http_client::HttpClient;
use language_model::{
    ApiKeyState, AuthenticateError, EnvVar, IconOrSvg, LanguageModel, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, LanguageModelToolChoice, OPEN_AI_PROVIDER_ID, OPEN_AI_PROVIDER_NAME,
    RateLimiter, env_var,
};
use menu;
use open_ai::{
    OPEN_AI_API_URL, ResponseStreamEvent,
    responses::{
        Request as ResponseRequest, RequestOptions as OpenAiResponseRequestOptions,
        ResponseTextConfig, ResponseTextVerbosity, StreamEvent as ResponsesStreamEvent,
        stream_response, stream_response_with_options,
    },
    stream_completion,
};
use settings::{OpenAiAvailableModel as AvailableModel, Settings, SettingsStore};
use std::sync::{Arc, LazyLock};
use strum::IntoEnumIterator;
use ui::{ButtonLink, ConfiguredApiCard, List, ListBulletItem, prelude::*};
use ui_input::InputField;
use util::ResultExt;

pub use open_ai::completion::{
    OpenAiEventMapper, OpenAiResponseEventMapper, collect_tiktoken_messages, count_open_ai_tokens,
    into_open_ai, into_open_ai_response,
};

use self::codex::{CodexAuthSession, PendingCodexOAuthFlow};

const PROVIDER_ID: LanguageModelProviderId = OPEN_AI_PROVIDER_ID;
const PROVIDER_NAME: LanguageModelProviderName = OPEN_AI_PROVIDER_NAME;

const API_KEY_ENV_VAR_NAME: &str = "OPENAI_API_KEY";
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OpenAiSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
}

pub struct OpenAiLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

pub struct State {
    api_key_state: ApiKeyState,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    codex_auth_session: Option<CodexAuthSession>,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.codex_auth_session.is_some() || self.api_key_state.has_key()
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = OpenAiLanguageModelProvider::api_url(cx);
        self.api_key_state.store(
            api_url,
            api_key,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        )
    }

    fn set_codex_auth_session(
        &mut self,
        session: Option<CodexAuthSession>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let credentials_provider = self.credentials_provider.clone();
        cx.spawn(async move |this, cx| {
            match session.as_ref() {
                Some(session) => {
                    codex::store_session(credentials_provider.as_ref(), session, cx).await?
                }
                None => codex::delete_session(credentials_provider.as_ref(), cx).await?,
            }

            this.update(cx, |this, cx| {
                this.codex_auth_session = session;
                cx.notify();
            })
        })
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = OpenAiLanguageModelProvider::api_url(cx);
        let api_key_task = self.api_key_state.load_if_needed(
            api_url,
            |this| &mut this.api_key_state,
            credentials_provider.clone(),
            cx,
        );

        cx.spawn(async move |this, cx| {
            if let Some(session) = codex::load_session(credentials_provider.as_ref(), cx).await? {
                this.update(cx, |this, cx| {
                    this.codex_auth_session = Some(session);
                    cx.notify();
                })?;
                return Ok(());
            }

            api_key_task.await
        })
    }

    fn clear_credentials(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = OpenAiLanguageModelProvider::api_url(cx);
        cx.spawn(async move |this, cx| {
            credentials_provider
                .delete_credentials(&api_url, cx)
                .await
                .log_err();
            credentials_provider
                .delete_credentials(codex::CREDENTIALS_KEY, cx)
                .await
                .log_err();

            this.update(cx, |this, cx| {
                this.api_key_state = ApiKeyState::new(api_url, (*API_KEY_ENV_VAR).clone());
                this.codex_auth_session = None;
                cx.notify();
            })
        })
    }
}

impl OpenAiLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>(|this: &mut State, cx| {
                let credentials_provider = this.credentials_provider.clone();
                let api_url = Self::api_url(cx);
                this.api_key_state.handle_url_change(
                    api_url,
                    |this| &mut this.api_key_state,
                    credentials_provider,
                    cx,
                );
                cx.notify();
            })
            .detach();
            State {
                api_key_state: ApiKeyState::new(Self::api_url(cx), (*API_KEY_ENV_VAR).clone()),
                credentials_provider,
                http_client: http_client.clone(),
                codex_auth_session: None,
            }
        });

        Self { http_client, state }
    }

    fn create_language_model(&self, model: open_ai::Model) -> Arc<dyn LanguageModel> {
        Arc::new(OpenAiLanguageModel {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }

    fn settings(cx: &App) -> &OpenAiSettings {
        &crate::AllLanguageModelSettings::get_global(cx).openai
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            open_ai::OPEN_AI_API_URL.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }
}

impl LanguageModelProviderState for OpenAiLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for OpenAiLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiOpenAi)
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(open_ai::Model::default()))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(open_ai::Model::default_fast()))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let mut models = BTreeMap::default();

        // Add base models from open_ai::Model::iter()
        for model in open_ai::Model::iter() {
            if !matches!(model, open_ai::Model::Custom { .. }) {
                models.insert(model.id().to_string(), model);
            }
        }

        // Override with available models from settings
        for model in &OpenAiLanguageModelProvider::settings(cx).available_models {
            models.insert(
                model.name.clone(),
                open_ai::Model::Custom {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens,
                    max_output_tokens: model.max_output_tokens,
                    max_completion_tokens: model.max_completion_tokens,
                    reasoning_effort: model.reasoning_effort,
                    supports_chat_completions: model.capabilities.chat_completions,
                },
            );
        }

        models
            .into_values()
            .map(|model| self.create_language_model(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(
        &self,
        _target_agent: language_model::ConfigurationViewTargetAgent,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyView {
        cx.new(|cx| ConfigurationView::new(self.state.clone(), window, cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state
            .update(cx, |state, cx| state.clear_credentials(cx))
    }
}

pub struct OpenAiLanguageModel {
    id: LanguageModelId,
    model: open_ai::Model,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

enum OpenAiAuthTransport {
    ApiKey {
        api_key: Arc<str>,
        api_url: SharedString,
    },
    CodexSubscription {
        access_token: String,
        account_id: String,
        api_url: &'static str,
    },
}

impl OpenAiLanguageModel {
    async fn auth_transport(
        state: Entity<State>,
        cx: &mut AsyncApp,
    ) -> Result<OpenAiAuthTransport, LanguageModelCompletionError> {
        let snapshot = state.read_with(cx, |state, cx| {
            let api_url = OpenAiLanguageModelProvider::api_url(cx);
            (
                state.codex_auth_session.clone(),
                state.api_key_state.key(&api_url),
                api_url,
                state.credentials_provider.clone(),
                state.http_client.clone(),
            )
        });

        let (codex_session, api_key, api_url, credentials_provider, http_client) = snapshot;

        if let Some(session) = codex_session {
            let session = if session.should_refresh() {
                let refreshed = codex::refresh_session(http_client, &session)
                    .await
                    .map_err(|error| LanguageModelCompletionError::AuthenticationError {
                        provider: PROVIDER_NAME,
                        message: format!("Failed to refresh ChatGPT subscription session: {error}"),
                    })?;
                codex::store_session(credentials_provider.as_ref(), &refreshed, cx)
                    .await
                    .log_err();
                state.update(cx, |state, cx| {
                    state.codex_auth_session = Some(refreshed.clone());
                    cx.notify();
                });
                refreshed
            } else {
                session
            };

            return Ok(OpenAiAuthTransport::CodexSubscription {
                access_token: session.access_token,
                account_id: session.account_id,
                api_url: "https://chatgpt.com/backend-api",
            });
        }

        let Some(api_key) = api_key else {
            return Err(LanguageModelCompletionError::NoApiKey {
                provider: PROVIDER_NAME,
            });
        };

        Ok(OpenAiAuthTransport::ApiKey { api_key, api_url })
    }

    fn uses_codex_subscription(&self, cx: &AsyncApp) -> bool {
        self.state
            .read_with(cx, |state, _| state.codex_auth_session.is_some())
    }

    fn stream_completion(
        &self,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>>>
    {
        let http_client = self.http_client.clone();
        let state = self.state.clone();
        let request_limiter = self.request_limiter.clone();
        let future = cx.spawn(async move |cx| {
            let transport = Self::auth_transport(state, cx).await?;
            request_limiter
                .stream(async move {
                    match transport {
                        OpenAiAuthTransport::ApiKey { api_key, api_url } => {
                            let response = stream_completion(
                                http_client.as_ref(),
                                PROVIDER_NAME.0.as_str(),
                                &api_url,
                                &api_key,
                                request,
                            )
                            .await?;
                            Ok(response)
                        }
                        OpenAiAuthTransport::CodexSubscription { .. } => {
                            Err(LanguageModelCompletionError::AuthenticationError {
                                provider: PROVIDER_NAME,
                                message:
                                    "ChatGPT subscription auth must use the Responses API path"
                                        .to_string(),
                            })
                        }
                    }
                })
                .await
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }

    fn stream_response(
        &self,
        request: ResponseRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<futures::stream::BoxStream<'static, Result<ResponsesStreamEvent>>>>
    {
        let http_client = self.http_client.clone();
        let state = self.state.clone();
        let request_limiter = self.request_limiter.clone();
        let future = cx.spawn(async move |cx| {
            let transport = Self::auth_transport(state, cx).await?;
            request_limiter
                .stream(async move {
                    match transport {
                        OpenAiAuthTransport::ApiKey { api_key, api_url } => {
                            let response = stream_response(
                                http_client.as_ref(),
                                PROVIDER_NAME.0.as_str(),
                                &api_url,
                                &api_key,
                                request,
                            )
                            .await?;
                            Ok(response)
                        }
                        OpenAiAuthTransport::CodexSubscription {
                            access_token,
                            account_id,
                            api_url,
                        } => {
                            let mut request = request;
                            let mut extra_headers = vec![
                                (
                                    "OpenAI-Beta".to_string(),
                                    "responses=experimental".to_string(),
                                ),
                                ("chatgpt-account-id".to_string(), account_id),
                                ("originator".to_string(), "codex_cli_rs".to_string()),
                                ("accept".to_string(), "text/event-stream".to_string()),
                            ];
                            if let Some(session_id) = request.prompt_cache_key.clone() {
                                extra_headers.push(("session_id".to_string(), session_id.clone()));
                                extra_headers.push(("conversation_id".to_string(), session_id));
                            }

                            request.store = Some(false);
                            request.max_output_tokens = None;
                            request.include = vec!["reasoning.encrypted_content".to_string()];
                            request.text = Some(ResponseTextConfig {
                                verbosity: ResponseTextVerbosity::Medium,
                            });

                            let response = stream_response_with_options(
                                http_client.as_ref(),
                                PROVIDER_NAME.0.as_str(),
                                api_url,
                                &access_token,
                                request,
                                OpenAiResponseRequestOptions {
                                    path: "/codex/responses",
                                    extra_headers,
                                },
                            )
                            .await?;
                            Ok(response)
                        }
                    }
                })
                .await
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl LanguageModel for OpenAiLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        use open_ai::Model;
        match &self.model {
            Model::FourOmniMini
            | Model::FourPointOneNano
            | Model::Five
            | Model::FiveCodex
            | Model::FiveMini
            | Model::FiveNano
            | Model::FivePointOne
            | Model::FivePointTwo
            | Model::FivePointTwoCodex
            | Model::FivePointThreeCodex
            | Model::FivePointFour
            | Model::FivePointFourMini
            | Model::FivePointFourPro
            | Model::O1
            | Model::O3 => true,
            Model::ThreePointFiveTurbo
            | Model::Four
            | Model::FourTurbo
            | Model::O3Mini
            | Model::Custom { .. } => false,
        }
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto => true,
            LanguageModelToolChoice::Any => true,
            LanguageModelToolChoice::None => true,
        }
    }

    fn supports_streaming_tools(&self) -> bool {
        true
    }

    fn supports_thinking(&self) -> bool {
        self.model.reasoning_effort().is_some()
    }

    fn supports_split_token_display(&self) -> bool {
        true
    }

    fn telemetry_id(&self) -> String {
        format!("openai/{}", self.model.id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        self.model.max_output_tokens()
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        let model = self.model.clone();
        cx.background_spawn(async move { count_open_ai_tokens(request, model) })
            .boxed()
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        if self.uses_codex_subscription(cx) {
            let request = into_open_ai_response(
                request,
                self.model.id(),
                self.model.supports_parallel_tool_calls(),
                self.model.supports_prompt_cache_key(),
                self.max_output_tokens(),
                self.model.reasoning_effort(),
            );
            let completions = self.stream_response(request, cx);
            async move {
                let mapper = OpenAiResponseEventMapper::new();
                Ok(mapper.map_stream(completions.await?).boxed())
            }
            .boxed()
        } else if self.model.supports_chat_completions() {
            let request = into_open_ai(
                request,
                self.model.id(),
                self.model.supports_parallel_tool_calls(),
                self.model.supports_prompt_cache_key(),
                self.max_output_tokens(),
                self.model.reasoning_effort(),
            );
            let completions = self.stream_completion(request, cx);
            async move {
                let mapper = OpenAiEventMapper::new();
                Ok(mapper.map_stream(completions.await?).boxed())
            }
            .boxed()
        } else {
            let request = into_open_ai_response(
                request,
                self.model.id(),
                self.model.supports_parallel_tool_calls(),
                self.model.supports_prompt_cache_key(),
                self.max_output_tokens(),
                self.model.reasoning_effort(),
            );
            let completions = self.stream_response(request, cx);
            async move {
                let mapper = OpenAiResponseEventMapper::new();
                Ok(mapper.map_stream(completions.await?).boxed())
            }
            .boxed()
        }
    }
}

struct ConfigurationView {
    api_key_editor: Entity<InputField>,
    state: Entity<State>,
    load_credentials_task: Option<Task<()>>,
    codex_login_task: Option<Task<()>>,
    codex_login_error: Option<SharedString>,
}

impl ConfigurationView {
    fn new(state: Entity<State>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let api_key_editor = cx.new(|cx| {
            InputField::new(
                window,
                cx,
                "sk-000000000000000000000000000000000000000000000000",
            )
        });

        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let load_credentials_task = Some(cx.spawn_in(window, {
            let state = state.clone();
            async move |this, cx| {
                if let Some(task) = Some(state.update(cx, |state, cx| state.authenticate(cx))) {
                    // We don't log an error, because "not signed in" is also an error.
                    let _ = task.await;
                }
                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            api_key_editor,
            state,
            load_credentials_task,
            codex_login_task: None,
            codex_login_error: None,
        }
    }

    fn save_api_key(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx).trim().to_string();
        if api_key.is_empty() {
            return;
        }

        // url changes can cause the editor to be displayed again
        self.api_key_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(Some(api_key), cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn start_codex_login(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.codex_login_task.is_some() {
            return;
        }

        self.codex_login_error = None;
        let flow = match codex::begin_oauth_flow() {
            Ok(flow) => flow,
            Err(error) => {
                self.codex_login_error = Some(error.to_string().into());
                cx.notify();
                return;
            }
        };

        cx.open_url(&flow.authorization_url);

        let state = self.state.clone();
        self.codex_login_task = Some(cx.spawn_in(window, async move |this, cx| {
            let result = finish_codex_login(flow, state, cx).await;
            this.update(cx, |this, cx| {
                this.codex_login_task = None;
                this.codex_login_error = result.err().map(|error| error.to_string().into());
                cx.notify();
            })
            .log_err();
        }));
        cx.notify();
    }

    fn reset_credentials(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.clear_credentials(cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn should_render_editor(&self, cx: &mut Context<Self>) -> bool {
        !self.state.read(cx).is_authenticated()
    }
}

async fn finish_codex_login(
    flow: PendingCodexOAuthFlow,
    state: Entity<State>,
    cx: &mut AsyncApp,
) -> Result<()> {
    let http_client = state.read_with(cx, |state, _| state.http_client.clone());
    let session = flow.finish(http_client).await?;
    let task = state.update(cx, |state, cx| {
        state.set_codex_auth_session(Some(session), cx)
    });
    task.await
}

impl Render for ConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (env_var_set, has_codex_auth, configured_card_label) = {
            let state = self.state.read(cx);
            let env_var_set = state.api_key_state.is_from_env_var();
            let has_codex_auth = state.codex_auth_session.is_some();
            let configured_card_label = if has_codex_auth {
                "ChatGPT Plus/Pro Codex subscription configured".to_string()
            } else if env_var_set {
                format!("API key set in {API_KEY_ENV_VAR_NAME} environment variable")
            } else {
                let api_url = OpenAiLanguageModelProvider::api_url(cx);
                if api_url == OPEN_AI_API_URL {
                    "API key configured".to_string()
                } else {
                    format!("API key configured for {}", api_url)
                }
            };
            (env_var_set, has_codex_auth, configured_card_label)
        };

        let api_key_section = if self.should_render_editor(cx) {
            v_flex()
                .on_action(cx.listener(Self::save_api_key))
                .gap_3()
                .child(Label::new("Use either an OpenAI API key or your ChatGPT Plus/Pro Codex subscription."))
                .child(
                    v_flex()
                        .gap_1()
                        .child(Label::new("ChatGPT Plus/Pro"))
                        .child(
                            Button::new("openai-codex-login", "Sign In With ChatGPT")
                                .disabled(self.codex_login_task.is_some())
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.start_codex_login(window, cx)
                                })),
                        )
                        .child(
                            Label::new(
                                "This uses the Codex OAuth flow locally and routes the provider through the ChatGPT Codex backend.",
                            )
                            .size(LabelSize::Small)
                            .color(Color::Muted),
                        )
                        .when_some(self.codex_login_error.clone(), |this, error| {
                            this.child(Label::new(error).size(LabelSize::Small).color(Color::Error))
                        })
                        .when(self.codex_login_task.is_some(), |this| {
                            this.child(
                                Label::new("Waiting for ChatGPT login in your browser…")
                                    .size(LabelSize::Small)
                                    .color(Color::Muted),
                            )
                        }),
                )
                .child(
                    div()
                        .pt_2()
                        .border_t_1()
                        .border_color(cx.theme().colors().border_variant),
                )
                .child(Label::new("API key"))
                .child(
                    List::new()
                        .child(
                            ListBulletItem::new("")
                                .child(Label::new("Create one by visiting"))
                                .child(ButtonLink::new("OpenAI's console", "https://platform.openai.com/api-keys"))
                        )
                        .child(
                            ListBulletItem::new("Ensure your OpenAI account has credits")
                        )
                        .child(
                            ListBulletItem::new("Paste your API key below and hit enter to start using the agent")
                        ),
                )
                .child(self.api_key_editor.clone())
                .child(
                    Label::new(format!(
                        "You can also set the {API_KEY_ENV_VAR_NAME} environment variable and restart Zed."
                    ))
                    .size(LabelSize::Small)
                    .color(Color::Muted),
                )
                .child(
                    Label::new(
                        "API keys are billed through the OpenAI platform. ChatGPT subscriptions only work through the button above.",
                    )
                    .size(LabelSize::Small).color(Color::Muted),
                )
                .into_any_element()
        } else {
            ConfiguredApiCard::new(configured_card_label)
                .disabled(env_var_set && !has_codex_auth)
                .on_click(cx.listener(|this, _, window, cx| this.reset_credentials(window, cx)))
                .when(env_var_set && !has_codex_auth, |this| {
                    this.tooltip_label(format!(
                        "To reset your API key, unset the {API_KEY_ENV_VAR_NAME} environment variable."
                    ))
                })
                .into_any_element()
        };

        let compatible_api_section = h_flex()
            .mt_1p5()
            .gap_0p5()
            .flex_wrap()
            .when(self.should_render_editor(cx), |this| {
                this.pt_1p5()
                    .border_t_1()
                    .border_color(cx.theme().colors().border_variant)
            })
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        Icon::new(IconName::Info)
                            .size(IconSize::XSmall)
                            .color(Color::Muted),
                    )
                    .child(Label::new("Zed also supports OpenAI-compatible models.")),
            )
            .child(
                Button::new("docs", "Learn More")
                    .end_icon(
                        Icon::new(IconName::ArrowUpRight)
                            .size(IconSize::Small)
                            .color(Color::Muted),
                    )
                    .on_click(move |_, _window, cx| {
                        cx.open_url("https://zed.dev/docs/ai/llm-providers#openai-api-compatible")
                    }),
            );

        if self.load_credentials_task.is_some() {
            div().child(Label::new("Loading credentials…")).into_any()
        } else {
            v_flex()
                .size_full()
                .child(api_key_section)
                .child(compatible_api_section)
                .into_any()
        }
    }
}
