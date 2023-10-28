---
title: Build a chatbot with custom data sources, powered by LlamaIndex
aliases: [ ]
tags: [sn, ]
author: Caroline Frasca
source: (https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)
published: 
clipped: 2023-10-28 09:41
topic: null

---
source: [Build a chatbot with custom data sources, powered by LlamaIndex](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)

# Build a chatbot with custom data sources, powered by LlamaIndex

> ## Excerpt
> Augment any LLM with your own data in 43 lines of code!

---
🎈

**TL;DR:**

Learn how LlamaIndex can enrich your LLM model with custom data sources through RAG pipelines. Build a chatbot app using LlamaIndex to augment GPT-3.5 with Streamlit documentation in just 43 lines of code.

So, you want to build a reliable chatbot using LLMs based on custom data sources?

Models like GPT are excellent at answering general questions from public data sources but aren't perfect. Accuracy takes a nose dive when you need to access domain expertise, recent data, or proprietary data sources.

Enhancing your LLM with custom data sources can feel overwhelming, especially when data is distributed across multiple (and siloed) applications, formats, and data stores.

**This is where [LlamaIndex](https://www.llamaindex.ai/?ref=blog.streamlit.io) comes in.**

LlamaIndex is a flexible framework that enables LLM applications to ingest, structure, access, and retrieve private data sources. The end result is that your model's responses will be more relevant and context-specific. Together with Streamlit, LlamaIndex empowers you to quickly create LLM-enabled apps enriched by your data. In fact, the LlamaIndex team used Streamlit to prototype and run experiments early in their journey, including their [initial proofs of concept](https://github.com/logan-markewich/llama_index_starter_pack?ref=blog.streamlit.io)!

In this post, we'll show you how to build a chatbot using LlamaIndex to augment GPT-3.5 with Streamlit documentation in four simple steps:

1.  Configure app secrets
2.  Install dependencies
3.  Build the app
4.  Deploy the app!

## What is LlamaIndex?

Before we get started, let's walk through the basics of [LlamaIndex](https://gpt-index.readthedocs.io/en/latest/index.html?ref=blog.streamlit.io).

Behind the scenes, LlamaIndex enriches your model with custom data sources through Retrieval Augmented Generation (RAG).

Overly simplified, this process generally consists of two stages:

1.  **An indexing stage.** LlamaIndex prepares the knowledge base by ingesting data and converting it into Documents. It parses metadata from those documents (text, relationships, and so on) into nodes and creates queryable indices from these chunks into the Knowledge Base.
2.  **A querying stage.** Relevant context is retrieved from the knowledge base to assist the model in responding to queries. The querying stage ensures the model can access data not included in its original training data.

[![rag-with-llamaindex-1](https://blog.streamlit.io/content/images/2023/08/rag-with-llamaindex-1.png#border)](https://blog.streamlit.io/content/images/2023/08/rag-with-llamaindex-1.png#border)

💬

**LlamaIndex for any level:** Tasks like enriching models with contextual data and constructing RAG pipelines have typically been reserved for experienced engineers, but LlamaIndex enables developers of all experience levels to approach this work. Whether you’re a beginner looking to get started in three lines of code, LlamaIndex unlocks the ability to supercharge your apps with both AI and your own data. For more complex applications, check out [Llama Lab](https://github.com/run-llama/llama-lab?ref=blog.streamlit.io).

No matter what your LLM data stack looks like, [LlamaIndex](https://www.llamaindex.ai/?ref=blog.streamlit.io) and [LlamaHub](https://llamahub.ai/?ref=blog.streamlit.io) likely already have an integration, and new integrations are added daily. [Integrations](https://gpt-index.readthedocs.io/en/stable/community/integrations.html?ref=blog.streamlit.io#integrations) with LLM providers, vector stores, data loaders, evaluation providers, and agent tools are already built.

LlamaIndex's [Chat Engines](https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/chat_engines/root.html?ref=blog.streamlit.io) pair nicely with Streamlit's chat elements, making building a contextually relevant chatbot fast and easy.

Let's unpack how to build one.

## How to build a custom chatbot using LlamaIndex

### In 43 lines of code, this app will:

-   Use LlamaIndex to load and index data. Specifically, we're using the markdown files that make up Streamlit's documentation (you can sub in your data if you want).
-   Create a chat UI with Streamlit's `st.chat_input` and `st.chat_message` methods
-   Store and update the chatbot's message history using the session state
-   Augment GPT-3.5 with the loaded, indexed data through LlamaIndex's chat engine interface so that the model provides relevant responses based on Streamlit's recent documentation

[![llamaindexgif](https://blog.streamlit.io/content/images/2023/08/llamaindexgif.gif#browser)](https://blog.streamlit.io/content/images/2023/08/llamaindexgif.gif#browser)

Try the app for yourself:

## 1\. Configure app secrets

This app will use GPT-3.5, so you'll also need an OpenAI API key. Follow our instructions [here](https://blog.streamlit.io/langchain-tutorial-1-build-an-llm-powered-app-in-18-lines-of-code/#step-1-get-an-openai-api-key) if you don't already have one.

Create a `secrets.toml` file with the following contents.

-   If you're using Git, be sure to add the name of this file to your `.gitignore` so you don't accidentally expose your API key.
-   If you plan to deploy this app on Streamlit Community Cloud, the following contents should be [added to your app's secrets via the Community Cloud modal](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management?ref=blog.streamlit.io#deploy-an-app-and-set-up-secrets).

  

```
openai_key = "<your OpenAI API key here>"
```

## 2\. Install dependencies

### 2.1. Local development

If you're working on your local machine, install dependencies using `pip`:

```
pip install streamlit openai llama-index nltk
```

### 2.2. Cloud development

If you're planning to deploy this app on Streamlit Community Cloud, create a `requirements.txt` file with the following contents:

```
streamlit
openai
llama-index
nltk
```

## 3\. Build the app

The [full app](https://github.com/carolinedlu/llamaindex-chat-with-streamlit-docs/blob/main/streamlit_app.py?ref=blog.streamlit.io) is only 43 lines of code. Let's break down each section.

### 3.1. Import libraries

Required Python libraries for this app: `streamlit`, `llama_index`, `openai`, and `nltk`.

```
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
```

### 3.2. Initialize message history

-   Set your OpenAI API key from the app's secrets.
-   Add a heading for your app.
-   Use [session state](https://docs.streamlit.io/library/api-reference/session-state?ref=blog.streamlit.io) to keep track of your chatbot's message history.
-   Initialize the value of `st.session_state.messages` to include the chatbot's starting message, such as, "Ask me a question about Streamlit's open-source Python library!"

  

```
openai.api_key = st.secrets.openai_key
st.header("Chat with the Streamlit docs 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]
```

### 3.3. Load and index data

Store your **Knowledge Base** files in a folder called `data` within the app. But before you begin…

Download the markdown files for Streamlit's documentation from the [`data`](https://github.com/carolinedlu/llamaindex-chat-with-streamlit-docs/tree/main/data?ref=blog.streamlit.io) demo app's GitHub repository folder. Or use [this link](http://github.com/carolinedlu/llamaindex-chat-with-streamlit-docs/zipball/main/?ref=blog.streamlit.io) to download a .zip file for the repo. Add the `data` folder to the root level of your app. Alternatively, add your data.

🎈

If you’re running your app locally, check out LlamaIndex’s library of data connectors, available via [LlamaHub](https://llamahub.ai/?ref=blog.streamlit.io), which makes it fast and easy to retrieve data from a variety of sources (including [GitHub repositories](https://llamahub.ai/l/github_repo?ref=blog.streamlit.io)).

Define a function called `load_data()`, which will:

-   Use LlamaIndex’s [`SimpleDirectoryReader`](https://gpt-index.readthedocs.io/en/stable/examples/data_connectors/simple_directory_reader.html?ref=blog.streamlit.io#simple-directory-reader) to passLlamaIndex's the folder where you’ve stored your data (in this case, it’s called `data` and sits at the base level of your repository).
-   `SimpleDirectoryReader` will select the appropriate file reader based on the extensions of the files in that directory (`.md` files for this example) and will load all files recursively from that directory when we call `reader.load_data()`.
-   Construct an instance of LlamaIndex’s `[ServiceContext](https://gpt-index.readthedocs.io/en/latest/core_modules/supporting_modules/service_context.html?ref=blog.streamlit.io)`, whichLlamaIndex'stion of resources used during a RAG pipeline's indexing and querying stages.
-   `ServiceContext` allows us to adjust settings such as the LLM and embedding model used.
-   Use LlamaIndex’s [`VectorStoreIndex`](https://gpt-index.readthedocs.io/en/stable/core_modules/data_modules/index/vector_store_guide.html?ref=blog.streamlit.io) to creaLlamaIndex'sory `SimpleVectorStore`, which will structure your data in a way that helps your model quickly retrieve context from your data. Learn more about LlamaIndex’s `Indices` [here](https://gpt-index.readthedocs.io/en/stable/core_modules/data_modules/index/root.html?ref=blog.streamlit.io#indexes). This function returns the `VectorStoreIndex` object.

This function is wrapped in Streamlit’s caching decorator `st.cache_resource` to minimize the number of times the data is loaded and indexed.

Finally, call the `load_data` function, designating its returned `VectorStoreIndex` object to be called `index`.

```
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
```

### 3.4. Create the chat engine

LlamaIndex offers several different modes of chat engines. It can be helpful to test each mode with questions specific to your knowledge base and use case, comparing the response generated by the model in each mode.

LlamaIndex has four different chat engines:

1.  **[Condense question engine](https://gpt-index.readthedocs.io/en/stable/examples/chat_engine/chat_engine_condense_question.html?ref=blog.streamlit.io)**: Always queries the knowledge base. Can have trouble with meta questions like “What did I previously ask you?”
2.  [**Context chat engin"**](https://gpt-index.readthedocs.io/en/stable/examples/chat_engine/chat_engine_context.html?ref=blog.streamlit.io): Always queries the knowledge base and uses retrieved text from the knowledge base as context for following queries. The retrieved context from previous queries can take up much of the available context for the current query.
3.  [**ReAct agent**](https://gpt-index.readthedocs.io/en/latest/examples/chat_engine/chat_engine_react.html?ref=blog.streamlit.io): Chooses whether to query the knowledge base or not. Its performance is more dependent on the quality of the LLM. You may need to coerce the chat engine to correctly choose whether to query the knowledge base.
4.  [**OpenAI agent**](https://gpt-index.readthedocs.io/en/latest/examples/chat_engine/chat_engine_openai.html?ref=blog.streamlit.io): Chooses whether to query the knowledge base or not—similar to ReAct agent mode, but uses [OpenAI’s built-in fuOpenAI'salling capabilities](https://platform.openai.com/docs/guides/gpt/function-calling?ref=blog.streamlit.io).

This example uses the **condense question mode** because it always queries the knowledge base (files from the Streamlit docs) when generating a response. This mode is optimal because you want the model to keep its answers specific to the features mentioned in Streamlit’s documentation.

```
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
```

### 3.5. Prompt for user input and display message history

-   Use Streamlit’s `st.chat_input` feature Streamlit'she user to enter a question.
-   Once the user has entered input, add that input to the message history by appending it `st.session_state.messages`.
-   Show the message history of the chatbot by iterating through the content associated with the “messages” key in the session state and displaying each message using `st.chat_message`.

  

```
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
```

### 3.6. Pass query to chat engine and display response

If the last message in the message history is not from the chatbot, pass the message content to the chat engine via `chat_engine.chat()`, write the response to the UI using `st.write` and `st.chat_message`, and add the chat engine’s response to the message history.

```
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
```

## 4\. Deploy the app!

After building the app, deploy it on Streamlit Community Cloud:

1.  Create a GitHub repository.
2.  Navigate to [Streamlit Community Cloud](https://share.streamlit.io/?ref=blog.streamlit.io), click `New app`, and pick the appropriate repository, branch, and file path.
3.  Hit `Deploy`.

### LlamaIndex helps prevent hallucinations

Now that you’ve built a Streamlit docs chatbot using up-to-date markdown files, how do these results compare the results to ChatGPT? GPT-3.5 and 4 have only been trained on data up to September 2021. They’re missing three years of new releases! Augmenting your LLM with LlamaIndex ensures higher accuracy of the response.

### What is Streamlit’s [experimental connection](https://docs.streamlit.io/library/api-reference/connections/st.experimental_connection?ref=blog.streamlit.io) feature?

[

![](https://blog.streamlit.io/content/images/2023/08/ChatGPT.png)



](https://blog.streamlit.io/content/images/2023/08/ChatGPT.png)

[

![](https://blog.streamlit.io/content/images/2023/08/LlamaIndexChatbot.png)



](https://blog.streamlit.io/content/images/2023/08/LlamaIndexChatbot.png)

[

![](https://blog.streamlit.io/content/images/2023/08/no.png)



](https://blog.streamlit.io/content/images/2023/08/no.png)

[

![](https://blog.streamlit.io/content/images/2023/08/yes.png)



](https://blog.streamlit.io/content/images/2023/08/yes.png)

## Wrapping up

You learned how the LlamaIndex framework can create RAG pipelines and supplement a model with your data.

You also built a chatbot app that uses LlamaIndex to augment GPT-3.5 in 43 lines of code. The Streamlit documentation can be substituted for any custom data source. The result is an app that yields far more accurate and up-to-date answers to questions about the Streamlit open-source Python library compared to ChatGPT or using GPT alone.

Check out our [LLM gallery](https://streamlit.io/gallery?category=llms&ref=blog.streamlit.io) for inspiration to build even more LLM-powered apps, and share your questions in the comments.

Happy Streamlit-ing! 🎈
