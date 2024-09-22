import express from 'express'
import { ChatOllama } from '@langchain/community/chat_models/ollama'
import { OpenAIEmbeddings } from '@langchain/openai'
import { fileURLToPath } from 'url'
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts'
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents'
import { createRetrievalChain } from 'langchain/chains/retrieval'
import { FaissStore } from '@langchain/community/vectorstores/faiss'
import { ChatOpenAI } from '@langchain/openai'
import dotenv from 'dotenv'
import { nanoid } from 'nanoid'
import path from 'path'
import { RecursiveCharacterTextSplitter, TokenTextSplitter } from 'langchain/text_splitter'
import { JSONLoader } from 'langchain/document_loaders/fs/json'

const app = express()
dotenv.config()

app.listen(3000, () => {
    console.log('Server running on port 3000')
})

app.use(express.urlencoded({ extended: false }))

const ollamaLlm = new ChatOllama({ baseUrl: 'http://localhost:11434/', model: 'llama3', temperature: 1 })
const ollamaLlm2 = new ChatOllama({ baseUrl: 'http://localhost:11434/', model: 'llama3.1', temperature: 1 })
const chatModel = new ChatOpenAI({ openAIApiKey: process.env.OpenAIKey, model: 'gpt-4o', temperature: 1 })

function createForms({ prompt = '', answer = '', openai = '', llama3 = '', llama31 = '' }) {
    return `
    <title>eBook</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" 
    integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
    <div class="container">
        <h1 class="text-center mt-3 mb-3">eBook</h1>
        <div class="card">
            <div class="card-header">Ask eBook</div>
            <div class="card-body">
                <form method="POST" id="myform" action="/">
                    <div class="mb-3">
                        <textarea rows="3" name="prompt" id="promptId" placeholder="Question" required class="form-control">${prompt}</textarea>
                    </div>
                    <div class="mb-3">
                        <input type="submit" class="btn btn-primary" value="Ask" />
                        <input type="checkbox" name="skipcache" checked> Skip Cache
                        <input type="checkbox" name="openai" ${openai ? 'checked' : ''}> Open AI
                        <input type="checkbox" name="llama3" ${llama3 ? 'checked' : ''}> LLAMA 3
                        <input type="checkbox" name="llama31" ${llama31 ? 'checked' : ''}> LLAMA 3.1
                    </div>
                    <div class="mb-3">
                        <span id="answertxt">${answer}</span>
                    </div>
                </form>
            </div>
        </div>
    </div>
    <script>
        document.getElementById("promptId").addEventListener("keydown", function(event) {
            if (event.which === 13  && !event.shiftKey && !event.repeat) {
                event.preventDefault();
                document.forms[0].submit();
            }
        });
    </script>`
}

async function processLLMResponse(model, prompt, retriever, question) {
    const documentChain = await createStuffDocumentsChain({ llm: model, prompt })
    const retrievalChain = await createRetrievalChain({ combineDocsChain: documentChain, retriever })
    const response = await retrievalChain.invoke({ input: question })
    return response.answer.replaceAll('\n', '<br/>')
}

async function getVectorStore(filename) {
    const loader = new JSONLoader(filename)
    const docs = await loader.load()
    const openaiEmbeddings = new OpenAIEmbeddings({ openAIApiKey: process.env.OpenAIKey })
    const vectorStore = await FaissStore.fromDocuments(docs, openaiEmbeddings)
    return vectorStore
}

app.get('/', (req, res) => {
    res.status(200).send(createForms({ openai: 'on', llama31: 'on' }))
})

app.post('/', async (req, res) => {
    const { prompt: question, skipcache, openai, llama3, llama31 } = req.body

    const openaiEmbeddings = new OpenAIEmbeddings({ openAIApiKey: process.env.OpenAIKey })
    const vectorStore = await getVectorStore('./test.json')
    const retriever = vectorStore.asRetriever()

    const systemTemplate = 'You are a teacher.'
    const humanTemplate = `
        Use the following pieces of context to answer the question at the end. 
        If can't find answer from context, just say that you don't know, don't try to make up an answer. Don't search the information from internet.
        You don't need to explain why you can't find answer.

        Context: {context}

        (You do not need to use these pieces of information if not relevant)
        
        Question: What's Video Section Id, video component id, and video timestamp to understand "{input}"? And tell why this video is most relevant.
        Give me the answer in valid json. Enclose value with quote in the Json.
    `

    const chatSystemPrompt = ChatPromptTemplate.fromMessages([
        ['system', systemTemplate],
        ['human', humanTemplate],
    ])

    let Lla3Response = llama3 === 'on' ? await processLLMResponse(ollamaLlm, chatSystemPrompt, retriever, question) : ''
    let llama31Response =
        llama31 === 'on' ? await processLLMResponse(ollamaLlm2, chatSystemPrompt, retriever, question) : ''
    let openAiResponse =
        openai === 'on' ? await processLLMResponse(chatModel, chatSystemPrompt, retriever, question) : ''

    const result = `<b>LLM Answer from context (OpenAI)</b><br/>${openAiResponse}<br/><b>LLM Answer from context (LLAMA 3)</b><br/>${Lla3Response}<br/><b>LLM Answer from context (LLAMA 3.1)</b><br/>${llama31Response}<br/>`

    res.status(200).send(createForms({ prompt: question, answer: result.trim(), openai, llama3, llama31 }))
})
