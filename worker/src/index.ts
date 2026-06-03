import { GoogleGenerativeAI } from '@google/generative-ai';

interface Env {
  GEMINI_API_KEY: string;
  MEMORY_BUCKET: R2Bucket;
}

interface ChatRequest {
  message: string;
  session_id?: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
    }

    // Health check
    if (path === '/health' && request.method === 'GET') {
      return new Response(JSON.stringify({ status: 'healthy' }), {
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
      });
    }

    // Chat endpoint
    if (path === '/chat' && request.method === 'POST') {
      try {
        const { message, session_id: sessionIdFromBody } = (await request.json()) as ChatRequest;
        const sessionId = sessionIdFromBody || crypto.randomUUID();

        // Load conversation from R2
        let conversation: any[] = [];
        const obj = await env.MEMORY_BUCKET.get(`${sessionId}.json`);
        if (obj) {
          conversation = await obj.json();
        }

        // Initialize Gemini
        const genAI = new GoogleGenerativeAI(env.GEMINI_API_KEY);
        const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

        // Build chat history for Gemini
        const chat = model.startChat({
          history: conversation.map(msg => ({
            role: msg.role === 'user' ? 'user' : 'model',
            parts: [{ text: msg.content }],
          })),
        });

        const result = await chat.sendMessage(message);
        const responseText = result.response.text();

        // Update conversation
        conversation.push({ role: 'user', content: message, timestamp: new Date().toISOString() });
        conversation.push({ role: 'model', content: responseText, timestamp: new Date().toISOString() });

        // Save to R2
        await env.MEMORY_BUCKET.put(`${sessionId}.json`, JSON.stringify(conversation), {
          httpMetadata: { contentType: 'application/json' },
        });

        return new Response(JSON.stringify({ response: responseText, session_id: sessionId }), {
          headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
        });
      } catch (err) {
        return new Response(JSON.stringify({ error: String(err) }), { status: 500 });
      }
    }

    return new Response('Not Found', { status: 404 });
  },
};