from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings


@lru_cache
def get_llm() -> BaseChatModel:
    """
    Provider-agnostic LLM factory. This is the fix for the exact failure
    mode hit earlier with gemini-1.5-flash: a retired model string blew up
    the whole app because it was hardcoded three layers deep. Here, the
    provider and model both come from config, and each branch is isolated
    so a retirement/outage on one provider is a one-line env change, not a
    code change or a redeploy of application logic.
    """
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            temperature=0.0,
            api_key=settings.anthropic_api_key,
        )

    if provider == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=0.0,
            google_api_key=settings.google_api_key,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
        "Expected one of: openai, anthropic, google_genai."
    )
