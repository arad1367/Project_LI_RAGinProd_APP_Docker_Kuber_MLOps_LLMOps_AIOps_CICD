import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";
const API_KEY = "dev-secret-key"; // DEV ONLY — see the security note below

export default function App() {
  const [messages, setMessages] = useState([]); // {role, content}
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  // A stable per-session id so the agent remembers this conversation (Phase 6 memory).
  const threadId = useRef("web-" + Math.random().toString(36).slice(2));
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;
    setInput("");
    // Add the user message and an empty assistant message we'll fill as it streams.
    setMessages((m) => [
      ...m,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);
    setLoading(true);

    try {
      const resp = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
        body: JSON.stringify({ question, thread_id: threadId.current }),
      });
      if (!resp.ok) throw new Error(`Server error ${resp.status}`);

      // Read the streaming text response chunk by chunk.
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        setMessages((m) => {
          const copy = [...m];
          copy[copy.length - 1] = { role: "assistant", content: acc };
          return copy;
        });
      }
    } catch (err) {
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = {
          role: "assistant",
          content: "⚠️ " + err.message,
        };
        return copy;
      });
    } finally {
      setLoading(false);
    }
  }

  async function uploadFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatus(`Uploading ${file.name}…`);
    const form = new FormData();
    form.append("file", file);
    try {
      const resp = await fetch(`${API_URL}/ingest`, {
        method: "POST",
        headers: { "X-API-Key": API_KEY }, // no Content-Type — the browser sets it for FormData
        body: form,
      });
      const data = await resp.json();
      setStatus(resp.ok ? `Ingested ${data.filename}` : "Upload failed");
    } catch (err) {
      setStatus("Upload error: " + err.message);
    } finally {
      e.target.value = ""; // let the same file be re-uploaded if needed
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>DocuMind</h1>
        <label className="upload">
          + Add document
          <input type="file" onChange={uploadFile} hidden />
        </label>
      </header>

      {status && <div className="status">{status}</div>}

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">Ask a question about your documents.</div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">{m.content || (loading ? "…" : "")}</div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask something… (Enter to send)"
          rows={1}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          {loading ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
