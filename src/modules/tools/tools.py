import datetime
import io
import json
import re
import tempfile
from collections import Counter

import arxiv
import pdfplumber
import pinecone
import requests
import streamlit as st
from langchain.chains import LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader, WikipediaLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import DuckDuckGoSearchResults, StructuredTool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.vectorstores import FAISS, Pinecone
from tenacity import retry, stop_after_attempt, wait_fixed

from ..ui import ui_config

ui = ui_config.create_ui_from_config()


llm_model = st.secrets["llm_model"]
langchain_verbose = st.secrets["langchain_verbose"]


# ##############################
# search_vector_database_tool
# ##############################
def vector_database_query_func_calling_chain():
    func_calling_json_schema = {
        "title": "get_querys_and_filters_to_search_vector_database",
        "description": "Extract the queries and filters for a vector database semantic search.",
        "type": "object",
        "properties": {
            "query": {
                "title": "Query",
                "description": "The queries extracted for a vector database semantic search in the format of a JSON object",
                "type": "string",
            },
            "created_at": {
                "title": "Date Filters",
                "description": 'Date extracted for a vector database semantic search, in MongoDB\'s query and projection operators, in format like {"$gte": 1609459200.0, "$lte": 1640908800.0}',
                "type": "string",
            },
        },
        "required": ["query"],
    }

    prompt_func_calling_msgs = [
        SystemMessage(
            content="You are a world class algorithm for extracting the queries and filters for a vector database semantic search. Make sure to answer in the correct structured format"
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]

    prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

    llm_func_calling = ChatOpenAI(model_name=llm_model, temperature=0, streaming=False)

    query_func_calling_chain = create_structured_output_chain(
        output_schema=func_calling_json_schema,
        llm=llm_func_calling,
        prompt=prompt_func_calling,
        verbose=langchain_verbose,
    )

    return query_func_calling_chain


def search_vector_database(query_input: str) -> list:
    """Use original query to semantic search in academic database."""

    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=st.secrets["pinecone_api_key"],
        environment=st.secrets["pinecone_environment"],
    )
    vectorstore = Pinecone.from_existing_index(
        index_name=st.secrets["pinecone_index"],
        embedding=embeddings,
    )

    query_response = vector_database_query_func_calling_chain().run(query_input)

    query = query_response.get("query")
    try:
        created_at = json.loads(query_response.get("created_at", None))
    except TypeError:
        created_at = None

    if created_at is not None:
        docs = vectorstore.similarity_search(
            query, k=16, filter={"created_at": created_at}
        )
    else:
        docs = vectorstore.similarity_search(query, k=16)

    docs_list = []
    for doc in docs:
        date = datetime.datetime.fromtimestamp(doc.metadata["created_at"])
        formatted_date = date.strftime("%Y-%m")  # Format date as 'YYYY-MM'
        source_entry = "[{}. {}. {}.]({})".format(
            doc.metadata["source_id"],
            doc.metadata["author"],
            formatted_date,
            doc.metadata["url"],
        )
        docs_list.append({"content": doc.page_content, "source": source_entry})

    return docs_list


search_vector_database_tool = StructuredTool.from_function(search_vector_database)


# ##############################
# search_internet_tool
# ##############################
def search_internet(query: str) -> list:
    """Search the internet for the up-to-date information."""
    search = DuckDuckGoSearchResults()
    results = search.run(query)

    pattern = r"\[snippet: (.*?), title: (.*?), link: (.*?)\]"
    matches = re.findall(pattern, results)

    docs = [
        {"snippet": match[0], "title": match[1], "link": match[2]} for match in matches
    ]

    docs_list = []

    for doc in docs:
        docs_list.append(
            {
                "content": doc["snippet"],
                "source": "[{}]({})".format(doc["title"], doc["link"]),
            }
        )

    return docs_list


search_internet_tool = StructuredTool.from_function(search_internet)


# ##############################
# search_arxiv_tool
# ##############################
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_file_to_stream(url):
    # 使用伪造的User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
    }

    response = requests.get(url, stream=True, headers=headers)

    # 检查状态码
    if response.status_code != 200:
        response.raise_for_status()  # 引发异常，如果有错误

    file_stream = io.BytesIO()

    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            file_stream.write(chunk)

    # 将文件指针重置到流的开头
    file_stream.seek(0)
    return file_stream


def parse_paper(pdf_stream):
    # logging.info("Parsing paper")
    pdf_obj = pdfplumber.open(pdf_stream)
    number_of_pages = len(pdf_obj.pages)
    # logging.info(f"Total number of pages: {number_of_pages}")
    full_text = ""
    ismisc = False
    for i in range(number_of_pages):
        page = pdf_obj.pages[i]
        if i == 0:
            isfirstpage = True
        else:
            isfirstpage = False

        page_text = []
        sentences = []
        processed_text = []

        def visitor_body(text, isfirstpage, x, top, bottom, fontSize, ismisc):
            # ignore header/footer
            if isfirstpage:
                if (top > 200 and bottom < 720) and (len(text.strip()) > 1):
                    sentences.append(
                        {
                            "fontsize": fontSize,
                            "text": " " + text.strip().replace("\x03", ""),
                            "x": x,
                            "y": top,
                        }
                    )
            else:  # not first page
                if (
                    (top > 70 and bottom < 720)
                    and (len(text.strip()) > 1)
                    and not ismisc
                ):  # main text region
                    sentences.append(
                        {
                            "fontsize": fontSize,
                            "text": " " + text.strip().replace("\x03", ""),
                            "x": x,
                            "y": top,
                        }
                    )
                elif (top > 70 and bottom < 720) and (len(text.strip()) > 1) and ismisc:
                    pass

        extracted_words = page.extract_words(
            x_tolerance=1,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
            horizontal_ltr=True,
            vertical_ttb=True,
            extra_attrs=["fontname", "size"],
            split_at_punctuation=False,
        )

        # Treat the first page, main text, and references differently, specifically targeted at headers
        # Define a list of keywords to ignore
        # Online is for Nauture papers
        keywords_for_misc = [
            "References",
            "REFERENCES",
            "Bibliography",
            "BIBLIOGRAPHY",
            "Acknowledgements",
            "ACKNOWLEDGEMENTS",
            "Acknowledgments",
            "ACKNOWLEDGMENTS",
            "参考文献",
            "致谢",
            "謝辞",
            "謝",
            "Online",
        ]

        prev_word_size = None
        prev_word_font = None
        # Loop through the extracted words
        for extracted_word in extracted_words:
            # Strip the text and remove any special characters
            text = extracted_word["text"].strip().replace("\x03", "")

            # Check if the text contains any of the keywords to ignore
            if any(keyword in text for keyword in keywords_for_misc) and (
                prev_word_size != extracted_word["size"]
                or prev_word_font != extracted_word["fontname"]
            ):
                ismisc = True

            prev_word_size = extracted_word["size"]
            prev_word_font = extracted_word["fontname"]

            # Call the visitor_body function with the relevant arguments
            visitor_body(
                text,
                isfirstpage,
                extracted_word["x0"],
                extracted_word["top"],
                extracted_word["bottom"],
                extracted_word["size"],
                ismisc,
            )

        if sentences:
            for sentence in sentences:
                page_text.append(sentence)

        blob_font_sizes = []
        blob_font_size = None
        blob_text = ""
        processed_text = ""
        tolerance = 1

        # Preprocessing for main text font size
        if page_text != []:
            if len(page_text) == 1:
                blob_font_sizes.append(page_text[0]["fontsize"])
            else:
                for t in page_text:
                    blob_font_sizes.append(t["fontsize"])
            blob_font_size = Counter(blob_font_sizes).most_common(1)[0][0]

        if page_text != []:
            if len(page_text) == 1:
                if (
                    blob_font_size - tolerance
                    <= page_text[0]["fontsize"]
                    <= blob_font_size + tolerance
                ):
                    processed_text += page_text[0]["text"]
                    # processed_text.append({"text": page_text[0]["text"], "page": i + 1})
            else:
                for t in range(len(page_text)):
                    if (
                        blob_font_size - tolerance
                        <= page_text[t]["fontsize"]
                        <= blob_font_size + tolerance
                    ):
                        blob_text += f"{page_text[t]['text']}"
                        if len(blob_text) >= 500:  # set the length of a data chunk
                            processed_text += blob_text
                            # processed_text.append({"text": blob_text, "page": i + 1})
                            blob_text = ""
                        elif t == len(page_text) - 1:  # last element
                            processed_text += blob_text
                            # processed_text.append({"text": blob_text, "page": i + 1})
            full_text += processed_text

    # logging.info("Done parsing paper")
    return full_text


def search_arxiv(query: str) -> list:
    """Search arxiv.org for results."""
    docs = arxiv.Search(
        query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance
    ).results()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=220, chunk_overlap=20
    )
    chunks = []

    for doc in docs:
        pdf_stream = download_file_to_stream(doc.pdf_url)

        page_content = parse_paper(pdf_stream)
        authors = ", ".join(str(author) for author in doc.authors)
        date = doc.published.strftime("%Y-%m")

        source = "[{}. {}. {}.]({})".format(
            authors,
            doc.title,
            date,
            doc.entry_id,
        )

        chunk = text_splitter.create_documents(
            [page_content], metadatas=[{"source": source}]
        )

        chunks.extend(chunk)

    embeddings = OpenAIEmbeddings()
    faiss_db = FAISS.from_documents(chunks, embeddings)

    result_docs = faiss_db.similarity_search(query, k=16)
    docs_list = []
    for doc in result_docs:
        source_entry = doc.metadata["source"]
        docs_list.append({"content": doc.page_content, "source": source_entry})

    return docs_list


search_arxiv_tool = StructuredTool.from_function(search_arxiv)


# ##############################
# search_wiki_tool
# ##############################
def wiki_query_func_calling_chain():
    func_calling_json_schema = {
        "title": "identify_query_language_to_search_Wikipedia",
        "description": "Accurately identifying the language of the query to search Wikipedia.",
        "type": "object",
        "properties": {
            "language": {
                "title": "Language",
                "description": "The accurate language of the query",
                "type": "string",
                "enum": [
                    "en",
                    "es",
                    "fr",
                    "de",
                    "ru",
                    "zh",
                    "pt",
                    "ar",
                    "it",
                    "ja",
                    "tr",
                    "id",
                    "simple",
                    "nl",
                    "pl",
                    "fa",
                    "he",
                    "vi",
                    "sv",
                    "ko",
                    "hi",
                    "uk",
                    "cs",
                    "ro",
                    "no",
                    "fi",
                    "hu",
                    "da",
                    "ca",
                    "th",
                    "bn",
                    "el",
                    "sr",
                    "bg",
                    "ms",
                    "hr",
                    "az",
                    "zh-yue",
                    "sk",
                    "sl",
                    "ta",
                    "arz",
                    "eo",
                    "sh",
                    "et",
                    "lt",
                    "ml",
                    "la",
                    "ur",
                    "af",
                    "mr",
                    "bs",
                    "sq",
                    "ka",
                    "eu",
                    "gl",
                    "hy",
                    "tl",
                    "be",
                    "kk",
                    "nn",
                    "ang",
                    "te",
                    "lv",
                    "ast",
                    "my",
                    "mk",
                    "ceb",
                    "sco",
                    "uz",
                    "als",
                    "zh-classical",
                    "is",
                    "mn",
                    "wuu",
                    "cy",
                    "kn",
                    "be-tarask",
                    "br",
                    "gu",
                    "an",
                    "bar",
                    "si",
                    "ne",
                    "sw",
                    "lb",
                    "zh-min-nan",
                    "jv",
                    "ckb",
                    "ga",
                    "war",
                    "ku",
                    "oc",
                    "nds",
                    "yi",
                    "ia",
                    "tt",
                    "fy",
                    "pa",
                    "azb",
                    "am",
                    "scn",
                    "lmo",
                    "gan",
                    "km",
                    "tg",
                    "ba",
                    "as",
                    "sa",
                    "ky",
                    "io",
                    "so",
                    "pnb",
                    "ce",
                    "vec",
                    "vo",
                    "mzn",
                    "or",
                    "cv",
                    "bh",
                    "pdc",
                    "hif",
                    "hak",
                    "mg",
                    "ht",
                    "ps",
                    "su",
                    "nap",
                    "qu",
                    "fo",
                    "bo",
                    "li",
                    "rue",
                    "se",
                    "nds-nl",
                    "gd",
                    "tk",
                    "yo",
                    "diq",
                    "pms",
                    "new",
                    "ace",
                    "vls",
                    "bat-smg",
                    "eml",
                    "cu",
                    "bpy",
                    "dv",
                    "hsb",
                    "sah",
                    "os",
                    "chr",
                    "sc",
                    "wa",
                    "szl",
                    "ha",
                    "ksh",
                    "bcl",
                    "nah",
                    "mt",
                    "co",
                    "ug",
                    "lad",
                    "cdo",
                    "pam",
                    "arc",
                    "crh",
                    "rm",
                    "zu",
                    "gv",
                    "frr",
                    "ab",
                    "got",
                    "iu",
                    "ie",
                    "xmf",
                    "cr",
                    "dsb",
                    "mi",
                    "gn",
                    "min",
                    "lo",
                    "sd",
                    "rmy",
                    "pcd",
                    "ilo",
                    "ext",
                    "sn",
                    "ig",
                    "nv",
                    "haw",
                    "csb",
                    "ay",
                    "jbo",
                    "frp",
                    "map-bms",
                    "lij",
                    "ch",
                    "vep",
                    "glk",
                    "tw",
                    "kw",
                    "bxr",
                    "wo",
                    "udm",
                    "av",
                    "pap",
                    "ee",
                    "cbk-zam",
                    "kv",
                    "fur",
                    "mhr",
                    "fiu-vro",
                    "bjn",
                    "roa-rup",
                    "gag",
                    "tpi",
                    "mai",
                    "stq",
                    "kab",
                    "bug",
                    "kl",
                    "nrm",
                    "mwl",
                    "bi",
                    "zea",
                    "ln",
                    "xh",
                    "myv",
                    "rw",
                    "nov",
                    "pfl",
                    "kaa",
                    "chy",
                    "roa-tara",
                    "pih",
                    "lfn",
                    "kg",
                    "bm",
                    "mrj",
                    "lez",
                    "za",
                    "om",
                    "ks",
                    "ny",
                    "krc",
                    "sm",
                    "st",
                    "pnt",
                    "dz",
                    "to",
                    "ary",
                    "tn",
                    "xal",
                    "gom",
                    "kbd",
                    "ts",
                    "rn",
                    "tet",
                    "mdf",
                    "ti",
                    "hyw",
                    "fj",
                    "tyv",
                    "ff",
                    "ki",
                    "ik",
                    "koi",
                    "lbe",
                    "jam",
                    "ss",
                    "lg",
                    "pag",
                    "tum",
                    "ve",
                    "ban",
                    "srn",
                    "ty",
                    "ltg",
                    "pi",
                    "sat",
                    "ady",
                    "olo",
                    "nso",
                    "sg",
                    "dty",
                    "din",
                    "tcy",
                    "gor",
                    "kbp",
                    "avk",
                    "lld",
                    "atj",
                    "inh",
                    "shn",
                    "nqo",
                    "mni",
                    "smn",
                    "mnw",
                    "dag",
                    "szy",
                    "gcr",
                    "awa",
                    "alt",
                    "shi",
                    "mad",
                    "skr",
                    "ami",
                    "trv",
                    "nia",
                    "tay",
                    "pwn",
                    "guw",
                    "pcm",
                    "kcg",
                    "blk",
                    "guc",
                    "anp",
                    "gur",
                    "fat",
                    "gpe",
                ],
            }
        },
        "required": ["language"],
    }

    prompt_func_calling_msgs = [
        SystemMessage(
            content="""You are a world class algorithm for accurately identifying the language of the query to search Wikipedia, strictly follow the language mapping: {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Russian": "ru", "Chinese": "zh", "Portuguese": "pt", "Arabic": "ar", "Italian": "it", "Japanese": "ja", "Turkish": "tr", "Indonesian": "id", "Simple English": "simple", "Dutch": "nl", "Polish": "pl", "Persian": "fa", "Hebrew": "he", "Vietnamese": "vi", "Swedish": "sv", "Korean": "ko", "Hindi": "hi", "Ukrainian": "uk", "Czech": "cs", "Romanian": "ro", "Norwegian": "no", "Finnish": "fi", "Hungarian": "hu", "Danish": "da", "Catalan": "ca", "Thai": "th", "Bangla": "bn", "Greek": "el", "Serbian": "sr", "Bulgarian": "bg", "Malay": "ms", "Croatian": "hr", "Azerbaijani": "az", "Cantonese": "zh-yue", "Slovak": "sk", "Slovenian": "sl", "Tamil": "ta", "Egyptian Arabic": "arz", "Esperanto": "eo", "Serbo-Croatian": "sh", "Estonian": "et", "Lithuanian": "lt", "Malayalam": "ml", "Latin": "la", "Urdu": "ur", "Afrikaans": "af", "Marathi": "mr", "Bosnian": "bs", "Albanian": "sq", "Georgian": "ka", "Basque": "eu", "Galician": "gl", "Armenian": "hy", "Tagalog": "tl", "Belarusian": "be", "Kazakh": "kk", "Norwegian Nynorsk": "nn", "Old English": "ang", "Telugu": "te", "Latvian": "lv", "Asturian": "ast", "Burmese": "my", "Macedonian": "mk", "Cebuano": "ceb", "Scots": "sco", "Uzbek": "uz", "Swiss German": "als", "Literary Chinese": "zh-classical", "Icelandic": "is", "Mongolian": "mn", "Wu Chinese": "wuu", "Welsh": "cy", "Kannada": "kn", "Belarusian (Taraškievica orthography)": "be-tarask", "Breton": "br", "Gujarati": "gu", "Aragonese": "an", "Bavarian": "bar", "Sinhala": "si", "Nepali": "ne", "Swahili": "sw", "Luxembourgish": "lb", "Min Nan Chinese": "zh-min-nan", "Javanese": "jv", "Central Kurdish": "ckb", "Irish": "ga", "Waray": "war", "Kurdish": "ku", "Occitan": "oc", "Low German": "nds", "Yiddish": "yi", "Interlingua": "ia", "Tatar": "tt", "Western Frisian": "fy", "Punjabi": "pa", "South Azerbaijani": "azb", "Amharic": "am", "Sicilian": "scn", "Lombard": "lmo", "Gan Chinese": "gan", "Khmer": "km", "Tajik": "tg", "Bashkir": "ba", "Assamese": "as", "Sanskrit": "sa", "Kyrgyz": "ky", "Ido": "io", "Somali": "so", "Western Punjabi": "pnb", "Chechen": "ce", "Venetian": "vec", "Volapük": "vo", "Mazanderani": "mzn", "Odia": "or", "Chuvash": "cv", "Bhojpuri": "bh", "Pennsylvania German": "pdc", "Fiji Hindi": "hif", "Hakka Chinese": "hak", "Malagasy": "mg", "Haitian Creole": "ht", "Pashto": "ps", "Sundanese": "su", "Neapolitan": "nap", "Quechua": "qu", "Faroese": "fo", "Tibetan": "bo", "Limburgish": "li", "Rusyn": "rue", "Northern Sami": "se", "Low Saxon": "nds-nl", "Scottish Gaelic": "gd", "Turkmen": "tk", "Yoruba": "yo", "Zazaki": "diq", "Piedmontese": "pms", "Newari": "new", "Achinese": "ace", "West Flemish": "vls", "Samogitian": "bat-smg", "Emiliano-Romagnolo": "eml", "Church Slavic": "cu", "Bishnupriya": "bpy", "Divehi": "dv", "Upper Sorbian": "hsb", "Yakut": "sah", "Ossetic": "os", "Cherokee": "chr", "Sardinian": "sc", "Walloon": "wa", "Silesian": "szl", "Hausa": "ha", "Colognian": "ksh", "Central Bikol": "bcl", "Nāhuatl": "nah", "Maltese": "mt", "Corsican": "co", "Uyghur": "ug", "Ladino": "lad", "Min Dong Chinese": "cdo", "Pampanga": "pam", "Aramaic": "arc", "Crimean Tatar": "crh", "Romansh": "rm", "Zulu": "zu", "Manx": "gv", "Northern Frisian": "frr", "Abkhazian": "ab", "Gothic": "got", "Inuktitut": "iu", "Interlingue": "ie", "Mingrelian": "xmf", "Cree": "cr", "Lower Sorbian": "dsb", "Māori": "mi", "Guarani": "gn", "Minangkabau": "min", "Lao": "lo", "Sindhi": "sd", "Vlax Romani": "rmy", "Picard": "pcd", "Iloko": "ilo", "Extremaduran": "ext", "Shona": "sn", "Igbo": "ig", "Navajo": "nv", "Hawaiian": "haw", "Kashubian": "csb", "Aymara": "ay", "Lojban": "jbo", "Arpitan": "frp", "Basa Banyumasan": "map-bms", "Ligurian": "lij", "Chamorro": "ch", "Veps": "vep", "Gilaki": "glk", "Twi": "tw", "Cornish": "kw", "Russia Buriat": "bxr", "Wolof": "wo", "Udmurt": "udm", "Avaric": "av", "Papiamento": "pap", "Ewe": "ee", "Chavacano": "cbk-zam", "Komi": "kv", "Friulian": "fur", "Eastern Mari": "mhr", "Võro": "fiu-vro", "Banjar": "bjn", "Aromanian": "roa-rup", "Gagauz": "gag", "Tok Pisin": "tpi", "Maithili": "mai", "Saterland Frisian": "stq", "Kabyle": "kab", "Buginese": "bug", "Kalaallisut": "kl", "Norman": "nrm", "Mirandese": "mwl", "Bislama": "bi", "Zeelandic": "zea", "Lingala": "ln", "Xhosa": "xh", "Erzya": "myv", "Kinyarwanda": "rw", "Novial": "nov", "Palatine German": "pfl", "Kara-Kalpak": "kaa", "Cheyenne": "chy", "Tarantino": "roa-tara", "Norfuk / Pitkern": "pih", "Lingua Franca Nova": "lfn", "Kongo": "kg", "Bambara": "bm", "Western Mari": "mrj", "Lezghian": "lez", "Zhuang": "za", "Oromo": "om", "Kashmiri": "ks", "Nyanja": "ny", "Karachay-Balkar": "krc", "Samoan": "sm", "Southern Sotho": "st", "Pontic": "pnt", "Dzongkha": "dz", "Tongan": "to", "Moroccan Arabic": "ary", "Tswana": "tn", "Kalmyk": "xal", "Goan Konkani": "gom", "Kabardian": "kbd", "Tsonga": "ts", "Rundi": "rn", "Tetum": "tet", "Moksha": "mdf", "Tigrinya": "ti", "Western Armenian": "hyw", "Fijian": "fj", "Tuvinian": "tyv", "Fula": "ff", "Kikuyu": "ki", "Inupiaq": "ik", "Komi-Permyak": "koi", "Lak": "lbe", "Jamaican Creole English": "jam", "Swati": "ss", "Ganda": "lg", "Pangasinan": "pag", "Tumbuka": "tum", "Venda": "ve", "Balinese": "ban", "Sranan Tongo": "srn", "Tahitian": "ty", "Latgalian": "ltg", "Pali": "pi", "Santali": "sat", "Adyghe": "ady", "Livvi-Karelian": "olo", "Northern Sotho": "nso", "Sango": "sg", "Doteli": "dty", "Dinka": "din", "Tulu": "tcy", "Gorontalo": "gor", "Kabiye": "kbp", "Kotava": "avk", "Ladin": "lld", "Atikamekw": "atj", "Ingush": "inh", "Shan": "shn", "N’Ko": "nqo", "Manipuri": "mni", "Inari Sami": "smn", "Mon": "mnw", "Dagbani": "dag", "Sakizaya": "szy", "Guianan Creole": "gcr", "Awadhi": "awa", "Southern Altai": "alt", "Tachelhit": "shi", "Madurese": "mad", "Saraiki": "skr", "Amis": "ami", "Taroko": "trv", "Nias": "nia", "Tayal": "tay", "Paiwan": "pwn", "Gun": "guw", "Nigerian Pidgin": "pcm", "Tyap": "kcg", "Pa"O": "blk", "Wayuu": "guc", "Angika": "anp", "Frafra": "gur", "Fanti": "fat", "Ghanaian Pidgin": "gpe"}"""
        ),
        HumanMessage(content="The query:"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]

    prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

    llm_func_calling = ChatOpenAI(model_name=llm_model, temperature=0, streaming=False)

    query_func_calling_chain = create_structured_output_chain(
        output_schema=func_calling_json_schema,
        llm=llm_func_calling,
        prompt=prompt_func_calling,
        verbose=langchain_verbose,
    )

    return query_func_calling_chain


def search_wiki(query: str) -> list:
    """Search Wikipedia for results."""
    language = wiki_query_func_calling_chain().run(query)["language"]
    docs = WikipediaLoader(
        query=query, lang=language, load_max_docs=3, load_all_available_meta=True
    ).load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=220, chunk_overlap=20
    )
    chunks = []

    for doc in docs:
        chunk = text_splitter.create_documents(
            [doc.page_content],
            metadatas=[
                {
                    "source": "[{}]({})".format(
                        doc.metadata["title"], doc.metadata["source"]
                    )
                }
            ],
        )
        chunks.extend(chunk)

    embeddings = OpenAIEmbeddings()
    faiss_db = FAISS.from_documents(chunks, embeddings)

    result_docs = faiss_db.similarity_search(query, k=16)

    docs_list = []

    for doc in result_docs:
        source_entry = doc.metadata["source"]
        docs_list.append({"content": doc.page_content, "source": source_entry})

    return docs_list


search_wiki_tool = StructuredTool.from_function(search_wiki)


# ##############################
# search_uploaded_docs_tool
# ##############################
def get_faiss_db(uploaded_files):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=220, chunk_overlap=20
    )
    chunks = []
    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=True) as fp:
                fp.write(uploaded_file.read())
                loader = UnstructuredFileLoader(file_path=fp.name)
                docs = loader.load()
                full_text = docs[0].page_content

            chunk = text_splitter.create_documents(
                texts=[full_text], metadatas=[{"source": uploaded_file.name}]
            )
            chunks.extend(chunk)
        except:
            pass
    if chunks != []:
        embeddings = OpenAIEmbeddings()
        faiss_db = FAISS.from_documents(chunks, embeddings)
    else:
        st.warning(ui.sidebar_file_uploader_error)
        st.stop()

    return chunks, faiss_db


def seach_uploaded_docs(query: str) -> list:
    """Semantic searches in uploaded documents."""
    docs = st.session_state["faiss_db"].similarity_search(query, k=16)
    docs_list = []
    for doc in docs:
        source_entry = doc.metadata["source"]
        docs_list.append({"content": doc.page_content, "source": source_entry})

    return docs_list


search_uploaded_docs_tool = StructuredTool.from_function(seach_uploaded_docs)


# ##############################
# dataframe_tool
# ##############################
# def get_dataframe(uploaded_files):


# ##############################
# calculation_tool
# ##############################
def calculation_tool():
    return PythonREPLTool()


# ##############################
# innovation_assessment_tool
# ##############################
def innovation_assessment(query: str) -> dict:
    """Get results for a detaied innovation assessment from upload files"""
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=st.secrets["pinecone_api_key"],
        environment=st.secrets["pinecone_environment"],
    )
    vectorstore = Pinecone.from_existing_index(
        index_name=st.secrets["pinecone_index"],
        embedding=embeddings,
    )
    # Initialize a list to store scores for each chunk
    docs_list = []
    chunk_scores = []
    for chunk in st.session_state["doc_chucks"]:
        docs = vectorstore.similarity_search_with_score(chunk.page_content, k=5)
        score = 0
        for doc in docs:
            date = datetime.datetime.fromtimestamp(doc[0].metadata["created_at"])
            formatted_date = date.strftime("%Y-%m")  # Format date as 'YYYY-MM'
            source_entry = "[{}. {}.]({})".format(
                doc[0].metadata["source_id"],
                formatted_date,
                doc[0].metadata["url"],
            )
            docs_list.append(
                {
                    "authors": doc[0].metadata["author"],
                    "source": source_entry,
                    "score": doc[1],
                }
            )
            score += doc[1]
        # Store the score and the chunk together
        chunk_scores.append({"score": score, "chunk": chunk.page_content})
    # Sort the list based on score in ascending order (lowest first)
    sorted_chunk_scores = sorted(chunk_scores, key=lambda x: x["score"])
    # Get the 16 chunks with the lowest scores along with their scores
    lowest_score_entries = sorted_chunk_scores[:16]
    lowest_score_chunks = [entry["chunk"] for entry in lowest_score_entries]
    # Sum the scores for the 16 chunks with the lowest scores
    # 去掉最高分（不要文本过短可能带来的极值）
    sum_of_lowest_scores = sum([entry["score"] for entry in lowest_score_entries])
    finally_score = round((100 * (90 - sum_of_lowest_scores) / 90), 2)

    # 使用一个字典来聚合每个 `source` 的总分和作者信息。
    source_scores = {}
    for doc in docs_list:
        source = doc["source"]
        if source not in source_scores:
            source_scores[source] = {"score": 0, "authors": doc["authors"]}
        else:
            source_scores[source]["score"] += doc["score"]

    # 将字典转换为一个列表。
    source_score_list = [
        {"source": key, "total_score": value["score"], "authors": value["authors"]}
        for key, value in source_scores.items()
    ]

    # 根据总分对列表进行排序。
    sorted_source_score_list = sorted(
        source_score_list, key=lambda x: x["total_score"], reverse=True
    )

    # 提取分数最高的10个 `source`，并带上authors的信息。
    top_10_sources_by_score = [
        {"authors": entry["authors"], "source": entry["source"]}
        for entry in sorted_source_score_list[:10]
    ]

    llm_chat = ChatOpenAI(
        model=llm_model,
        temperature=0,
        streaming=False,
        verbose=langchain_verbose,
    )
    prompt_potential_innovation_msgs = [
        SystemMessage(
            content="""Summarize potential innovations from text snipets. Use bullet points if a better expression effect can be achieved."""
        ),
        HumanMessage(content="The text snipets:"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]

    prompt_potential_innovation = ChatPromptTemplate(
        messages=prompt_potential_innovation_msgs
    )
    chain = LLMChain(llm=llm_chat, prompt=prompt_potential_innovation)
    potential_innovations = chain.run(lowest_score_chunks)

    result = {
        "potential_innovations": potential_innovations,
        "innovation_score": finally_score,
        "recommended_reviewers_and_their_relevant_published_articles": top_10_sources_by_score,
    }

    return result


innovation_assessment_tool = StructuredTool.from_function(
    innovation_assessment, return_direct=False
)
