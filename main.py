import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from qdrant_client import QdrantClient, models
from typing import List, Optional
import json
import uuid
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import traceback

# Load environment variables
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "kinza-saeed-collection"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIM = 768

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
if not QDRANT_HOST:
    raise ValueError("QDRANT_HOST not found in environment variables")

# --- FastAPI App ---
app = FastAPI(title="RAG Chatbot API", description="FastAPI service for a RAG Chatbot using Gemini and Qdrant.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
openai_client: AsyncOpenAI | None = None
qdrant_client: QdrantClient | None = None


@app.on_event("startup")
async def startup_event():
    global openai_client, qdrant_client
    openai_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY, 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    if QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://"):
        qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    else:
        qdrant_client = QdrantClient(host=QDRANT_HOST, api_key=QDRANT_API_KEY)
    
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' exists with {collection_info.points_count} points")
    except Exception as e:
        print(f"Collection not found. Creating '{COLLECTION_NAME}'...")
        try:
            create_qdrant_collection()
            print(f"Collection '{COLLECTION_NAME}' created successfully")
        except Exception as create_error:
            print(f"Error creating collection: {create_error}")


# --- Pydantic Models ---
class Document(BaseModel):
    id: str | int
    content: str
    metadata: dict = {}  # Can include: title, source, url, author, date, etc.


class BulkDocuments(BaseModel):
    documents: List[Document]


class ChatMessage(BaseModel):
    role: str
    content: str


class SourceInfo(BaseModel):
    """Enhanced source information"""
    id: str
    title: Optional[str] = None
    source_type: Optional[str] = None
    url: Optional[str] = None
    relevance_score: float
    content_preview: str


class ChatRequest(BaseModel):
    query: str
    top_k: int = 4
    stream: bool = False


class ChatWithHistoryRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []
    top_k: int = 4
    stream: bool = False


class DeleteDocumentRequest(BaseModel):
    id: str | int


# --- Helper Functions ---
async def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the given text using Gemini's model."""
    global openai_client
    response = await openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def create_qdrant_collection():
    """Creates/recreates a Qdrant collection."""
    global qdrant_client
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
    )
    return {"message": f"Collection '{COLLECTION_NAME}' created or recreated."}


async def index_document(doc: Document):
    global qdrant_client
    embedding = await get_embedding(doc.content)
    
    doc_id = str(doc.id)
    
    try:
        point_id = uuid.UUID(doc_id)
    except ValueError:
        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
    
    # Store enhanced metadata
    payload = {
        "content": doc.content,
        "original_id": doc_id,
        **doc.metadata  # This will include title, source, url, etc.
    }
    
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(point_id),
                payload=payload,
                vector=embedding,
            )
        ],
        wait=True
    )
    return {"message": f"Document '{doc_id}' indexed successfully.", "point_id": str(point_id)}


async def index_bulk_documents(documents: List[Document]):
    """Index multiple documents at once with parallel embedding generation."""
    global qdrant_client
    
    async def process_document(doc: Document):
        embedding = await get_embedding(doc.content)
        
        doc_id = str(doc.id)
        
        try:
            point_id = uuid.UUID(doc_id)
        except ValueError:
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
        
        payload = {
            "content": doc.content,
            "original_id": doc_id,
            **doc.metadata
        }
        
        return models.PointStruct(
            id=str(point_id),
            payload=payload,
            vector=embedding,
        )
    
    points = []
    batch_size = 10 
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_points = await asyncio.gather(*[process_document(doc) for doc in batch])
        points.extend(batch_points)
        print(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
    
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )
    return {"message": f"{len(documents)} documents indexed successfully."}


def delete_document(doc_id: str | int):
    """Delete a document by ID."""
    global qdrant_client
    
    doc_id_str = str(doc_id)
    
    try:
        point_id = uuid.UUID(doc_id_str)
    except ValueError:
        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id_str)
    
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(
            points=[str(point_id)],
        ),
        wait=True
    )
    return {"message": f"Document '{doc_id}' deleted successfully."}


def list_documents(limit: int = 100, offset: Optional[str] = None):
    """List documents with pagination."""
    global qdrant_client
    
    result = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    
    documents = [
        {
            "id": point.payload.get("original_id", str(point.id)),
            "point_id": str(point.id),
            "content": point.payload.get("content", ""),
            "metadata": {k: v for k, v in point.payload.items() if k not in ["content", "original_id"]}
        }
        for point in result[0]
    ]
    
    return {
        "documents": documents,
        "count": len(documents),
        "next_offset": result[1]
    }


def clear_collection():
    """Delete and recreate the collection."""
    global qdrant_client
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME, timeout=10)
    return create_qdrant_collection()


def format_sources(search_points) -> List[SourceInfo]:
    """Format search results into structured source information."""
    sources = []
    seen_ids = set()
    
    for point in search_points:
        if not point.payload:
            continue
            
        original_id = point.payload.get("original_id", str(point.id))
        
        # Avoid duplicate sources
        if original_id in seen_ids:
            continue
        seen_ids.add(original_id)
        
        # Extract metadata
        title = point.payload.get("title") or point.payload.get("name") or f"Document {original_id}"
        source_type = point.payload.get("source_type") or point.payload.get("type") or "document"
        url = point.payload.get("url") or point.payload.get("link")
        content = point.payload.get("content", "")
        
        # Create content preview (first 150 chars)
        preview = content[:150] + "..." if len(content) > 150 else content
        
        source_info = SourceInfo(
            id=original_id,
            title=title,
            source_type=source_type,
            url=url,
            relevance_score=round(point.score, 3),
            content_preview=preview
        )
        sources.append(source_info)
    
    return sources


async def rag_query(query_text: str, top_k: int = 4, history: List[ChatMessage] = None) -> dict:
    """Main RAG query function with enhanced source formatting."""
    global qdrant_client, openai_client
    
    # 1. Generate query embedding
    query_embedding = await get_embedding(query_text)
    
    # 2. Check collection status
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        return {
            "response": "The knowledge base is empty. Please upload documents first before asking questions.", 
            "sources": []
        }
    
    # 3. Search in Qdrant (use query_points for scores)
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
        score_threshold=0.3  # Optional: filter low relevance results
    )
    
    # 4. Extract context and format sources
    context_docs = []
    for point in search_result.points:
        if point.payload and point.payload.get("content"):
            context_docs.append(point.payload.get("content", ""))
    
    if not context_docs:
        return {
            "response": "I couldn't find any relevant information to answer your question. Please try rephrasing or upload more documents.", 
            "sources": []
        }
    
    # Format sources with all metadata
    sources = format_sources(search_result.points)
    context = "\n\n".join(context_docs)
    
    # 5. Build prompt messages with source citation instruction
    system_prompt = """You are a helpful assistant. Use the provided context to answer questions accurately and concisely. 
    
Important guidelines:
- Base your answer ONLY on the provided context
- If the context doesn't contain enough information, clearly state this
- Be specific and cite relevant details from the context
- Keep answers concise but informative"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        *(({"role": msg.role, "content": msg.content} for msg in history) if history else []),
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
    ]
    
    # 6. Generate response
    response = await openai_client.chat.completions.create(
        model="gemini-2.5-flash",  # Latest stable Gemini model
        messages=messages,
        max_tokens=800,
        temperature=0.7
    )
    
    answer = response.choices[0].message.content
    
    return {
        "response": answer, 
        "sources": [source.dict() for source in sources]
    }


async def rag_query_stream(query_text: str, top_k: int = 4, history: List[ChatMessage] = None):
    """Stream responses from the RAG query with sources sent first."""
    global qdrant_client, openai_client
    
    try:
        # 1. Generate query embedding
        query_embedding = await get_embedding(query_text)
        
        # 2. Check collection status
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        
        if collection_info.points_count == 0:
            yield "data: " + json.dumps({
                "type": "error",
                "content": "The knowledge base is empty. Please upload documents first."
            }) + "\n\n"
            yield "data: [DONE]\n\n"
            return
        
        # 3. Search in Qdrant
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            score_threshold=0.3
        )
        
        # 4. Extract context
        context_docs = [
            point.payload.get("content", "")  
            for point in search_result.points  
            if point.payload and point.payload.get("content")
        ]
        
        if not context_docs:
            yield "data: " + json.dumps({
                "type": "error",
                "content": "I couldn't find any relevant information to answer your question."
            }) + "\n\n"
            yield "data: [DONE]\n\n"
            return
        
        # Send sources first
        sources = format_sources(search_result.points)
        yield "data: " + json.dumps({
            "type": "sources",
            "sources": [source.dict() for source in sources]
        }) + "\n\n"
        
        context = "\n\n".join(context_docs)
        
        # 5. Build prompt messages
        system_prompt = """You are a helpful assistant. Use the provided context to answer questions accurately and concisely. 
        
Important guidelines:
- Base your answer ONLY on the provided context
- If the context doesn't contain enough information, clearly state this
- Be specific and cite relevant details from the context
- Keep answers concise but informative"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            *(({"role": msg.role, "content": msg.content} for msg in history) if history else []),
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"}
        ]
        
        # 6. Stream response
        stream = await openai_client.chat.completions.create(
            model="gemini-2.5-flash",  # Latest stable Gemini model
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield "data: " + json.dumps({
                    "type": "content",
                    "content": chunk.choices[0].delta.content
                }) + "\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"Error in rag_query_stream: {str(e)}")
        traceback.print_exc()
        yield "data: " + json.dumps({
            "type": "error",
            "content": f"Error: {str(e)}"
        }) + "\n\n"
        yield "data: [DONE]\n\n"


def check_qdrant_health():
    """Check if Qdrant is accessible."""
    global qdrant_client
    try:
        collections = qdrant_client.get_collections()
        return {
            "status": "healthy",
            "collections": [col.name for col in collections.collections]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "RAG Chatbot is running."}


@app.get("/health")
async def health_check():
    """Check the health of the service, Qdrant connection, and embedding model."""
    qdrant_status = check_qdrant_health()
    
    embedding_test = None
    try:
        test_embedding = await get_embedding("test")
        embedding_test = {"status": "ok", "dimension": len(test_embedding)}
    except Exception as e:
        embedding_test = {"status": "error", "error": str(e)}
    
    return {
        "service": "healthy",
        "qdrant": qdrant_status,
        "embedding": embedding_test
    }


@app.get("/test-search")
async def test_search():
    """Test the vector search functionality in Qdrant."""
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        
        if collection_info.points_count == 0:
            return {
                "status": "empty",
                "message": "No documents in collection. Upload some first.",
                "points_count": 0
            }
        
        test_embedding = await get_embedding("test query for RAG system")
        
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=test_embedding,
            limit=3,
            with_payload=True
        )
        
        return {
            "status": "success",
            "points_count": collection_info.points_count,
            "search_results": len(search_result),
            "results": [
                {
                    "score": hit.score,
                    "content_preview": hit.payload.get("content", "")[:100] + "..."
                }
                for hit in search_result
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/create-collection")
async def api_create_collection():
    try:
        return create_qdrant_collection()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/upload-document")
async def api_upload_document(doc: Document):
    try:
        return await index_document(doc)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Document upload failed: {str(e)}")


@app.post("/upload-documents-bulk")
async def api_upload_documents_bulk(bulk_docs: BulkDocuments):
    """Upload multiple documents at once."""
    try:
        return await index_bulk_documents(bulk_docs.documents)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Bulk upload failed: {str(e)}")


@app.delete("/delete-document")
async def api_delete_document(request: DeleteDocumentRequest):
    """Delete a specific document by ID."""
    try:
        return delete_document(request.id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Document deletion failed: {str(e)}")


@app.get("/list-documents")
async def api_list_documents(limit: int = 100, offset: Optional[str] = None):
    """List documents in the collection with pagination."""
    try:
        return list_documents(limit, offset)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Listing documents failed: {str(e)}")


@app.post("/clear-collection")
async def api_clear_collection():
    """Delete all documents and recreate the collection."""
    try:
        return clear_collection()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Collection clearing failed: {str(e)}")


@app.post("/chat")
async def api_chat(request: ChatRequest):
    """Simple RAG chat endpoint without history."""
    try:
        if request.stream:
            return StreamingResponse(
                rag_query_stream(request.query, request.top_k),
                media_type="text/event-stream"
            )
        else:
            result = await rag_query(request.query, request.top_k)
            return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Chat failed: {str(e)}"
        )


@app.post("/chat-with-history")
async def api_chat_with_history(request: ChatWithHistoryRequest):
    """RAG chat with conversation history support."""
    try:
        if request.stream:
            return StreamingResponse(
                rag_query_stream(request.query, request.top_k, request.history),
                media_type="text/event-stream"
            )
        else:
            result = await rag_query(request.query, request.top_k, request.history)
            return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Chat with history failed: {str(e)}"
        )