Current pipeline

We've now added:

                  DOCUMENT
                     ↓
                TextLoader
                     ↓
               Document Object
                ↙          ↘
        page_content     metadata

Our overall project currently has two separate pipelines:

Generation pipeline
User
 ↓
Prompt
 ↓
LLM
 ↓
Parser
 ↓
Answer
  
Document pipeline
File
 ↓
Loader
 ↓
Document
 ↓
Text + Metadata

Eventually we'll connect them:

                    DOCUMENT
                       ↓
                     Loader
                       ↓
                    Chunking
                       ↓
                   Embeddings
                       ↓
                 Vector Database
                       ↓
                    Retriever
                       ↓
USER → Query → Relevant Context
                       ↓
                     Prompt
                       ↓
                      LLM
                       ↓
                    Answer