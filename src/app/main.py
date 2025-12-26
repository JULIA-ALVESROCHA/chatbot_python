"""
Main FastAPI app for BGO Chatbot.

- Inicializa settings (Pydantic)
- Inicializa vectorstore (FAISS) na startup
- Registra rotas (src.app.api.v1.chat)
- Configura CORS, logging e handlers básicos
"""
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse 
from fastapi.middleware.cors import CORSMiddleware

from src.app.api.v1 import chat
from src.app.core.config import settings
from src.rag_pipeline.retrieval.vectorstore import init_vectorstore

logger = logging.getLogger("bgo_chatbot")
logging.basicConfig(level=logging.INFO)

def create_app() -> FastAPI:
    app = FastAPI(
        title="BGO Chatbot API",
        description="Backend API for the Brazilian Geography Olympiad Chatbot (RAG pipeline)",
        version="0.1.0",
    )

    # CORS (ajuste origns - when the chatbot is publish online)
    app.add_middleware(
        CORSMiddleware,
        allow_origins= settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["POST", "OPTIONS"],
        allow_headers=["*"],
    )

    #Include API router
    app.include_router(chat.router, prefix="/api/v1")

    #Health & readiness endpoints
    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok"}
    
    @app.get("/ready", tags=["health"])
    async def ready():
        #check if the vectorstore is loading
        try:
            # lazy check: se init_vectorstore levantou exceção na startup,
            # a app provavelmente não está pronta — então respondemos conforme.
            return {"ready": True}
        except Exception:
            return {"ready": False}

    # Global exception handler (retorna JSON com mensagem curta)
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc:Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail":"Internal server error"}
        )
    
    # Startup: inicializa recursos pesados
    @app.on_event("startup")
    async def on_startup():
        logger.info("Starting BGO Chatbot API...")
        # 1️⃣ Validar OpenAI key
        if not settings.openai_api_key:
            logger.error("OPENAI_API_KEY não definido.")
        else:
            logger.info("OPENAI_API_KEY carregada.")
        
        # 2️⃣ Inicializar FAISS (OBRIGATÓRIO)
        try:
            logger.info(
                "Inicializando vectorstore (FAISS) em: %s",
                settings.faiss_index_path
            )
            init_vectorstore(settings.faiss_index_path)
            logger.info("Vectorstore inicializado com sucesso.")
        except FileNotFoundError as e:
            logger.error("FAISS index não encontrado: %s", e)
            raise RuntimeError("FAISS index obrigatório para rodar o chatbot")
        except Exception as e:
            logger.exception("Erro ao inicializar vectorstore")
            raise
    
    return app

app = create_app()