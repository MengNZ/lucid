import { useState, useRef, useEffect } from 'react'
import './App.css'

// 与后端 api.py 的 NODE_LABELS 对应
const NODES = [
  { key: 'preprocess_node', label: '理解输入' },
  { key: 'signals_node', label: '分析行为信号' },
  { key: 'supervisor_node', label: '规划分析路径' },
  { key: 'compression_node', label: '压缩历史上下文' },
  { key: 'evidence_node', label: '收集证据' },
  { key: 'alternative_explanation_node', label: '生成替代解释' },
  { key: 'information_symmetry_node', label: '检查信息对称性' },
  { key: 'contradictory_registration_node', label: '汇总矛盾' },
  { key: 'summary_node', label: '生成诊断' },
]

const freshNodes = () => NODES.map((n) => ({ ...n, status: 'pending' }))

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [threadId, setThreadId] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send() {
    const msg = input.trim()
    if (!msg || busy) return
    setInput('')
    setBusy(true)

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: msg },
      { role: 'assistant', content: '', nodes: freshNodes(), done: false, error: null },
    ])

    try {
      const resp = await fetch('/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_msg: msg, thread_id: threadId }),
      })

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      const apply = (event) => {
        setMessages((prev) => {
          const next = [...prev]
          const idx = next.length - 1
          const last = next[idx]
          // 不可变更新：每个分支都复制新对象，绝不原地改 last。
          // 否则 React StrictMode 会双重调用 updater，导致 token 每个字 append 两遍。
          if (event.type === 'node_start') {
            next[idx] = {
              ...last,
              nodes: last.nodes.map((n) =>
                n.key === event.node ? { ...n, status: 'running' } : n
              ),
            }
          } else if (event.type === 'node_end') {
            next[idx] = {
              ...last,
              nodes: last.nodes.map((n) =>
                n.key === event.node ? { ...n, status: 'done' } : n
              ),
            }
          } else if (event.type === 'token') {
            next[idx] = { ...last, content: last.content + event.content }
          } else if (event.type === 'done') {
            next[idx] = { ...last, done: true }
            setThreadId(event.thread_id)
          } else if (event.type === 'error') {
            next[idx] = { ...last, error: event.message, done: true }
          }
          return next
        })
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        let idx
        while ((idx = buffer.indexOf('\n\n')) !== -1) {
          const raw = buffer.slice(0, idx)
          buffer = buffer.slice(idx + 2)
          const line = raw.split('\n').find((l) => l.startsWith('data: '))
          if (line) apply(JSON.parse(line.slice(6)))
        }
      }
    } catch (e) {
      setMessages((prev) => {
        const next = [...prev]
        const last = next[next.length - 1]
        last.error = String(e)
        last.done = true
        return next
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Lucid</h1>
        <p>恋爱关系认知校准 · 把「猜对方在想什么」变成可检查的流程</p>
      </header>

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty">
            <p>粘贴你们的聊天记录，或描述你的关系困惑</p>
            <p className="hint">例：她上周拒绝了我两次邀约，说太忙。我觉得她可能不喜欢我。</p>
          </div>
        )}

        {messages.map((m, i) =>
          m.role === 'user' ? (
            <div key={i} className="msg user">{m.content}</div>
          ) : (
            <div key={i} className="msg assistant">
              <Progress nodes={m.nodes} done={m.done} />
              {m.content && <div className="answer">{renderContent(m.content)}</div>}
              {m.error && <div className="error">分析出错：{m.error}</div>}
              {!m.done && !m.content && <div className="loading">分析中…</div>}
            </div>
          )
        )}
        <div ref={endRef} />
      </main>

      <footer className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
          placeholder="粘贴聊天记录，或补充信息…（Enter 发送，Shift+Enter 换行）"
          rows={3}
          disabled={busy}
        />
        <button onClick={send} disabled={busy || !input.trim()}>
          {busy ? '分析中…' : '发送'}
        </button>
      </footer>
    </div>
  )
}

function renderContent(text) {
  // 极简 markdown：只处理 **加粗**，其余保留换行（pre-wrap）
  const parts = String(text).split(/\*\*(.+?)\*\*/g)
  return parts.map((p, i) => (i % 2 === 1 ? <b key={i}>{p}</b> : p))
}

function Progress({ nodes, done }) {
  if (!nodes || (done && nodes.every((n) => n.status === 'done'))) return null
  const active = nodes.find((n) => n.status === 'running')
  const finished = nodes.filter((n) => n.status === 'done').length
  return (
    <div className="progress">
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${(finished / nodes.length) * 100}%` }} />
      </div>
      <div className="progress-text">
        {done ? '分析完成' : active ? `正在${active.label}…` : '准备分析…'}
      </div>
      <div className="node-dots">
        {nodes.map((n) => (
          <span key={n.key} className={`dot ${n.status}`} title={n.label} />
        ))}
      </div>
    </div>
  )
}

export default App
