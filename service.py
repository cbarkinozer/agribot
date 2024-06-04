import os
import PyPDF2
from docx import Document
from fastapi import UploadFile
from user import User
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import pickle
from datetime import datetime
import io
from dotenv import load_dotenv
import threading
from langdetect import detect
from sentence_transformers import SentenceTransformer
import torch


async def upload_documents(user: User, files: list[UploadFile]) -> tuple[str, int]:
    text = await _extract_text_from_document(files)
    chunks = await _chunk_text(text)
    await _create_embeddings_and_save(user, chunks)
    return "Document is uploaded successfully.", 200

async def _extract_text_from_document(files: list[UploadFile]) -> str:
    text = ""
    for file in files:
        byte_object = await file.read()
        file_name = file.filename
        file_extension = os.path.splitext(file_name)[1]
        if file_extension == '.txt':
            text += byte_object.decode('utf-8')
        elif file_extension == '.pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(byte_object))
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                text += page.extract_text()
        elif file_extension == '.docx':
            doc = Document(io.BytesIO(byte_object))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
    return text

async def _chunk_text(text: str) -> list[str]:
    chunks = None
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=10,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

async def _create_embeddings_and_save(user: User, chunks: any) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=user.embedder)
    pkl_name = os.path.join(user.username + ".pkl")
    vector_store = FAISS.from_texts(chunks, embeddings, metadatas=[{"source": f"{pkl_name}:{i}"} for i in range(len(chunks))])
    with open(pkl_name, "wb") as f:
        pickle.dump(vector_store, f)
    return vector_store


models = {}
class BackgroundTasks(threading.Thread):
    def run(self):
        try:
            model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            encoder = SentenceTransformer(model, cache_folder='./encoder')
            models["model"] = model
            models["encoder"] = encoder
        except Exception as e:
            print(f"Error loading model: {e}")

thread = BackgroundTasks()
thread.start()

functions = {
    "weather": {
        "en_description":"How's the weather? Will it rain? Will it be cold? Weather. Weather information. Moisture. The soil dried up.",
        "tr_description":"Hava nasıl? Yağacak mı? Soğuk mu olacak? Hava durumu. Hava durumu bilgileri. Nem. Toprak kurudu."
    },
    "sickness":{
        "en_description":"Bug. It was infested. My plants are not growing. My plants turned yellow. My plants are dying. It does not produce crops. I can't get a crop.",
        "tr_description":"Böcek. Böceklendi. Kurudu. Çürüdü. Sarardı. Büyümüyor. Mahsul alamadım. Bitkilerim ölüyor. Mahsül vermiyor. Mahsül alamıyorum."
    },
    "rag": {
        "en_description":"Aquaponics. Hydroponics. Soilless agriculture. Aquaculture. Vertical farming.",
        "tr_description":"Akuaponik. Hidroponik. Topraksız tarım. Akuakültür. Dikey tarım."
    }
}

async def ask_question(user: User, question: str, api_key: str) -> tuple[str, int]:
    
    encoder = models["encoder"]
    question_embedding = encoder.encode(question)
    request_language = detect(question)

    if request_language == "tr":
        description = "tr_description"
    else:
        description = "en_description"
    
    similarities = {}
    for function, data in functions.items():
        description_embedding = encoder.encode(data[description])
        cosine_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        request_embedding_tensor = torch.tensor(question_embedding)
        description_embedding_tensor = torch.tensor(description_embedding)
        similarity = cosine_sim(request_embedding_tensor, description_embedding_tensor)
        similarities[function] = similarity.item()
    
    max_similarity = max(similarities.values())
    max_similarity_function = max(similarities, key=similarities.get)

    if max_similarity < 0.3:
        if request_language=="tr":
            return "İsteğinizi anlayamadım. İsteğinizi farklı bir şekilde ifade etmeyi deneyebilir misiniz?", 400
        else:
            return "I couldn't understand your request. Can you try expressing your request in a different way?", 400

    print(f"[DEBUG] Selected Function: {max_similarity_function}")

    if max_similarity_function == "weather":
        answer = await _weather()
    elif max_similarity_function == "sickness":
        answer = await _sickness()
    elif max_similarity_function == "rag":
        answer = await _rag(user, question, api_key)
    else:
        return "Routed Function Name Exception", 500
    
    system_message = """
    Sen tarımla alakalı insanlara yardımcı olan bir sohbet asistanısın.
    Senin görevin kullanıcının sana sorduğu sorulara sana verilen bilgileri kullanarak cevap vermek.
    Cevaplarının tonu dostane ve nötrdür.
    Cevabının en az birkaç cümle olmalıdır.
    Sana verilen bilgilerin dışında bilgiler kullanma.
    <Örnek 1>:
    <Human>:
    <Soru>:
    Mahsül alamıyorum, ekinlerim ölüyor. Sorun ne olabilir?
    <Bilgi>:
    Maalesef hastalıklar konusunda yardımcı olamıyoruz. Lütfen bir uzmana danışın.
    <AI>:
    Bu durumun birçok farklı nedeni olabilir.
    Maalesef hastalıklar konusunda size yardımcı olamıyorum.
    Bu konuda bir uzmana danışmanızı öneririm.
    Uzman, ekinlerinizin durumunu yerinde değerlendirerek en doğru teşhisi koyup size uygun çözümler sunacaktır.

    <Örnek 2>:
    <Human>:
    <Soru>:
    Yağmur ne zaman yağacak tarlam çok kurudu?
    <Bilgi>:
    Hava durumu parçalı bulutlu 30°C, Yağış: 0%, Nem: 28%, Rüzgar: 18 km/s
    <AI>:
    Maalesef, mevcut hava durumu bilgilerine göre yağmur beklenmiyor.
    Hava parçalı bulutlu, sıcaklık 30°C, nem oranı %28 ve rüzgar hızı 18 km/s.
    Bu koşullarda yağış olasılığı %0 görünüyor.
    Tarlanızı sulamak için alternatif yöntemler düşünmeniz gerekebilir.
    Yakın zamanda bir yağış tahmini almak için hava durumu raporlarını düzenli olarak kontrol etmekte fayda var.

    <Örnek 3>:
    <Human>:
    <Soru>:
    Nasıl topraksız tarım yapabilirim?
    <Bilgi>:
    Metinde böyle bir bilgi bulunmamaktadır.
    <AI>:
    Maalesef, elimdeki bilgilerde bu sorunuzun cevabı bulunmuyor.
    İsterseniz sorunuzu farklı bir şekilde dile getirin belki o şekilde istediğiniz bilgiyle alakalı kısmı sizin için bulabilirim.

    Şimdi sıra sende:
    """

    chat = ChatGoogleGenerativeAI(model=user.llm, temperature=0, max_output_tokens=256, top_k = 40, top_p = 0.8)
    final_answer = chat.invoke(f" Sistem Mesajı: {system_message} <Soru>: {question} <Bilgi>: {answer}")
    final_answer = final_answer.content

    await _log(user=user, question=question, system_message=system_message, selected_function= max_similarity_function, answer= answer, final_answer = final_answer)

    return final_answer, 200

async def _weather() -> str:
    return "Hava durumu parçalı bulutlu 30°C, Yağış: 0%, Nem: 28%, Rüzgar: 18 km/s"

async def _sickness() -> str:
    return "Maalesef hastalıklar konusunda yardımcı olamıyoruz. Lütfen bir uzmana danışın."

async def _rag(user: User, question: str, api_key: str) -> tuple[str, int]:
    vector_store = await _get_vector_file(user.username)
    if vector_store is None:
        return "Document not found.", 400
    
    if api_key is not None:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        is_loaded = load_dotenv()
        if is_loaded == False:
            return "API key not found.", 400
        
    llm = ChatGoogleGenerativeAI(model=user.llm, temperature=0, max_output_tokens=256, top_k = 40, top_p = 0.8)
    docs = vector_store.similarity_search(question)
    retrieved_chunks = docs[0].page_content + docs[1].page_content + docs[2].page_content
    system_message="Figure out the answer of the question by the given information pieces. ALWAYS answer with the language of the question."
    prompt = system_message + "Question: " + question + " Context: " + retrieved_chunks
    try:
        response = llm.invoke(prompt)
    except Exception:
        return "Wrong API key.", 400
    answer = response.content + "  **<Most Related Chunk>**  " + retrieved_chunks
    print(f"[DEBUG] RAG Results: {answer}")
    return answer, 200


async def _get_vector_file(username: str)-> any:
    with open(username+".pkl", "rb") as f:
        vector_store = pickle.load(f)
    return vector_store

async def _log(user: User, question: str, system_message: str, selected_function: str, answer: str, final_answer: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    username = user.username
    llm = user.llm
    embedder = user.embedder

    log_message = f"\n{timestamp}, Username: {username}, Question: {question}, LLM: {llm}, Embedder: {embedder}, System Message: {system_message}, Selected Function: {selected_function}, Answer: {answer}, Final Answer: {final_answer}\n"
    with open("log.txt", "a", encoding="utf-8") as file:
        file.write("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        file.write(log_message)
        file.write("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")