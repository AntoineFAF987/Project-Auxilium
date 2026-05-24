// API pour la gestion des chats persistés côté serveur

// Helper pour fetch avec timeout
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = 10000): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('Délai d\'attente dépassé - le serveur ne répond pas');
    }
    throw err;
  }
}

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export type Chat = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export async function fetchChats(idToken: string): Promise<Chat[]> {
  try {
    const r = await fetchWithTimeout(`${process.env.NEXT_PUBLIC_API_URL}/chats`, {
      headers: { Authorization: `Bearer ${idToken}` },
      cache: "no-store",
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const data = await r.json();
    return Array.isArray(data?.items) ? data.items : [];
  } catch (err: any) {
    if (err.message && err.message.includes("HTTP")) {
      throw err;
    }
    throw new Error(`Erreur réseau: ${err.message || "Impossible de contacter le serveur"}`);
  }
}

export async function createChat(params: { title?: string; id?: string }, idToken: string): Promise<Chat> {
  try {
    const r = await fetchWithTimeout(`${process.env.NEXT_PUBLIC_API_URL}/chats`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${idToken}`,
      },
      body: JSON.stringify({ title: params.title, id: params.id }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const data = await r.json();
    return {
      id: String(data?.id),
      title: String(data?.title || "Nouveau chat"),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
  } catch (err: any) {
    if (err.message && err.message.includes("HTTP")) {
      throw err;
    }
    throw new Error(`Erreur réseau: ${err.message || "Impossible de contacter le serveur"}`);
  }
}

export async function fetchChatMessages(chatId: string, idToken: string): Promise<ChatMessage[]> {
  try {
    const r = await fetchWithTimeout(`${process.env.NEXT_PUBLIC_API_URL}/chats/${chatId}/messages`, {
      headers: { Authorization: `Bearer ${idToken}` },
      cache: "no-store",
    });
    if (!r.ok) {
      const errorText = await r.text();
      throw new Error(`HTTP ${r.status}: ${errorText}`);
    }
    const data = await r.json();
    const items = Array.isArray(data?.items) ? data.items : [];
    return items.map((m: any) => ({
      id: String(m.id || crypto.randomUUID()),
      role: (m.role === "assistant" ? "assistant" : "user") as "user" | "assistant",
      content: String(m.content || ""),
    }));
  } catch (err: any) {
    if (err.message && err.message.includes("HTTP")) {
      throw err;
    }
    throw new Error(`Erreur réseau: ${err.message || "Impossible de contacter le serveur"}`);
  }
}

export async function renameChat(chatId: string, title: string, idToken: string): Promise<void> {
  try {
    const r = await fetchWithTimeout(`${process.env.NEXT_PUBLIC_API_URL}/chats/${chatId}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${idToken}`,
      },
      body: JSON.stringify({ title }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  } catch (err: any) {
    if (err.message && err.message.includes("HTTP")) {
      throw err;
    }
    throw new Error(`Erreur réseau: ${err.message || "Impossible de contacter le serveur"}`);
  }
}

export async function deleteChat(chatId: string, idToken: string): Promise<void> {
  try {
    const r = await fetchWithTimeout(`${process.env.NEXT_PUBLIC_API_URL}/chats/${chatId}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${idToken}` },
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  } catch (err: any) {
    if (err.message && err.message.includes("HTTP")) {
      throw err;
    }
    throw new Error(`Erreur réseau: ${err.message || "Impossible de contacter le serveur"}`);
  }
}
