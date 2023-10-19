user_query = """Produce a detailed review on the topic of Dynamic Material Flow Analysis (DMFA). You must keep all details, and ensure that well-structured and organized, with a logical flow of ideas. It must be longer than 500 words."""


user_query = """According to the above response, add more details about methodology. You must keep all details, and ensure that well-structured and organized, with a logical flow ofideas. It must be longer than 800 words."""


prompt_template = """You must:
        keep the previous response from the following history if any;
        leverage the folloing information from uploaded files and knowlege base, and your own knowledge, provide an updated response to the "{query}";
        ensure response logical, clear, well-organized, and critically analyzed; 
        give in-text citations where relevant in Author-Date mode, NOT in Numeric mode;
        ensure response as detailed as possible;
        ensure response longer than 1000 words.
        
        HISTORY:
        "{history}".

        UPLOADED INFO:
        "{uploaded_docs}".

        KNOWLEDGE BASE:
        "{pinecone_docs}".

        You must not:
        include any duplicate or redundant information."""