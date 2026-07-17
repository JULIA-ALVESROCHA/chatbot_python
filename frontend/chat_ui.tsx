import React, { useState, useEffect, useRef } from "react"

// Icon Components
const SendIcon = () => (
    <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
    >
        <line x1="22" y1="2" x2="11" y2="13"></line>
        <polygon points="22 2 15 22 11 13 2 9"></polygon>
    </svg>
)

const XIcon = () => (
    <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
    >
        <line x1="18" y1="6" x2="6" y2="18"></line>
        <line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>
)

const MessageCircleIcon = ({ size = 24 }) => (
    <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
    >
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
    </svg>
)

const PaperclipIcon = () => (
    <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
    >
        <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
    </svg>
)

interface Message {
    id: number
    text: string
    sender: "user" | "bot"
    timestamp: Date
    sources?: Array<{
        document: string
        section: string
    }>
}

export default function ChatbotFramer() {
    const apiUrl = "https://chatbotpython-production.up.railway.app/api/v1/chat"
    const botName = "GeoLUME"
    const initialMessage =
        "Olá! Posso responder perguntas com base nos documentos oficiais da OBG."

    // Updated color scheme - Soft Blue theme
    const buttonColor = "#4A90E2" // Soft blue for main button
    const botIconColor = "#E8F4FF" // Very light blue for bot icon background
    const userIconColor = "#4A90E2" // Soft blue for user icon
    const sendButtonColor = "#10B981" // Green for send button

    const [isOpen, setIsOpen] = useState(false)
    const [messages, setMessages] = useState<Message[]>([
        {
            id: 1,
            text: initialMessage,
            sender: "bot",
            timestamp: new Date(),
            sources: [],
        },
    ])
    const [inputValue, setInputValue] = useState("")
    const [isTyping, setIsTyping] = useState(false)
    const [sessionId, setSessionId] = useState<string | null>(null)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (!sessionId) {
            setSessionId(
                `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
            )
        }
    }, [])

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [messages])

    const sendMessage = async (text: string) => {
        if (!text.trim()) return

        const userMessage: Message = {
            id: Date.now(),
            text: text.trim(),
            sender: "user",
            timestamp: new Date(),
            sources: [],
        }

        setMessages((prev) => [...prev, userMessage])
        setInputValue("")
        setIsTyping(true)

        try {
            const response = await fetch(apiUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    question: text.trim(),
                    session_id: sessionId,
                }),
            })

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            const data = await response.json()

            const botMessage: Message = {
                id: Date.now() + 1,
                text:
                    data.answer ||
                    "Não encontrei essa informação nos documentos oficiais.",
                sources: data.sources || [],
                sender: "bot",
                timestamp: new Date(),
            }

            setMessages((prev) => [...prev, botMessage])
        } catch (error) {
            console.error("Error sending message:", error)

            const errorMessage: Message = {
                id: Date.now() + 1,
                text: "❌ Erro ao conectar com o servidor. Verifique se o backend está rodando.",
                sender: "bot",
                timestamp: new Date(),
                sources: [],
            }

            setMessages((prev) => [...prev, errorMessage])
        } finally {
            setIsTyping(false)
        }
    }

    const handleSendMessage = () => {
        sendMessage(inputValue)
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault()
            handleSendMessage()
        }
    }

    const handleQuickReply = (text: string) => {
        sendMessage(text)
    }

    const quickReplies = [
        "Qual é o prazo de inscrição?",
        "Como funciona a avaliação?",
        "Posso participar individualmente?",
        "Onde encontro o regulamento?",
        "Como acessar materiais de estudo?",
    ]
    const hasUserMessage = messages.some((message) => message.sender === "user")

    return (
        <>
            {/* Chat Widget Button */}
            {!isOpen && (
                <button
                    onClick={() => setIsOpen(true)}
                    style={{
                        position: "fixed",
                        bottom: "24px",
                        right: "24px",
                        width: "64px",
                        height: "64px",
                        background: buttonColor,
                        borderRadius: "50%",
                        border: "none",
                        boxShadow: "0 10px 25px rgba(74, 144, 226, 0.3)",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        zIndex: 999999,
                        transition: "transform 0.2s ease",
                        color: "white",
                    }}
                    onMouseEnter={(e) =>
                        (e.currentTarget.style.transform = "scale(1.1)")
                    }
                    onMouseLeave={(e) =>
                        (e.currentTarget.style.transform = "scale(1)")
                    }
                >
                    <MessageCircleIcon size={32} />
                </button>
            )}

            {/* Chat Window */}
            {isOpen && (
                <div
                    style={{
                        position: "fixed",
                        bottom: "24px",
                        right: "24px",
                        width: "400px",
                        maxWidth: "calc(100vw - 48px)",
                        height: "700px",
                        maxHeight: "calc(100vh - 48px)",
                        backgroundColor: "white",
                        borderRadius: "24px",
                        boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
                        display: "flex",
                        flexDirection: "column",
                        zIndex: 999999,
                        overflow: "hidden",
                    }}
                >
                    {/* Header */}
                    <div
                        style={{
                            backgroundColor: "white",
                            padding: "16px",
                            borderBottom: "1px solid #E5E7EB",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                        }}
                    >
                        <div
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "12px",
                            }}
                        >
                            <div
                                style={{
                                    width: "40px",
                                    height: "40px",
                                    background: botIconColor,
                                    borderRadius: "12px",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    boxShadow:
                                        "0 4px 12px rgba(74, 144, 226, 0.15)",
                                    color: buttonColor,
                                }}
                            >
                                <MessageCircleIcon size={24} />
                            </div>
                            <h3
                                style={{
                                    margin: 0,
                                    fontSize: "18px",
                                    fontWeight: "600",
                                    color: "#111827",
                                }}
                            >
                                {botName}
                            </h3>
                        </div>
                        <button
                            onClick={() => setIsOpen(false)}
                            style={{
                                background: "transparent",
                                border: "none",
                                padding: "8px",
                                cursor: "pointer",
                                borderRadius: "8px",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                transition: "background 0.2s",
                            }}
                            onMouseEnter={(e) =>
                                (e.currentTarget.style.background = "#F3F4F6")
                            }
                            onMouseLeave={(e) =>
                                (e.currentTarget.style.background =
                                    "transparent")
                            }
                        >
                            <XIcon />
                        </button>
                    </div>

                    {/* Date Separator */}
                    <div
                        style={{
                            padding: "12px",
                            display: "flex",
                            justifyContent: "center",
                        }}
                    >
                        <span
                            style={{
                                fontSize: "12px",
                                color: "#9CA3AF",
                                backgroundColor: "#F9FAFB",
                                padding: "4px 12px",
                                borderRadius: "12px",
                            }}
                        >
                            Today
                        </span>
                    </div>

                    {/* Messages Area */}
                    <div
                        style={{
                            flex: 1,
                            overflowY: "auto",
                            padding: "16px",
                            display: "flex",
                            flexDirection: "column",
                            gap: "12px",
                        }}
                    >
                        {messages.map((message) => (
                            <div
                                key={message.id}
                                style={{
                                    display: "flex",
                                    alignItems: "flex-end",
                                    gap: "8px",
                                    justifyContent:
                                        message.sender === "user"
                                            ? "flex-end"
                                            : "flex-start",
                                }}
                            >
                                {message.sender === "bot" && (
                                    <div
                                        style={{
                                            width: "32px",
                                            height: "32px",
                                            background: botIconColor,
                                            borderRadius: "12px",
                                            display: "flex",
                                            alignItems: "center",
                                            justifyContent: "center",
                                            flexShrink: 0,
                                            boxShadow:
                                                "0 2px 8px rgba(74, 144, 226, 0.15)",
                                            color: buttonColor,
                                        }}
                                    >
                                        <MessageCircleIcon size={18} />
                                    </div>
                                )}

                                <div
                                    style={{
                                        maxWidth: "75%",
                                        borderRadius: "16px",
                                        padding: "12px 16px",
                                        backgroundColor:
                                            message.sender === "user"
                                                ? buttonColor
                                                : "#F3F4F6",
                                        color:
                                            message.sender === "user"
                                                ? "white"
                                                : "#1F2937",
                                        borderBottomRightRadius:
                                            message.sender === "user"
                                                ? "4px"
                                                : "16px",
                                        borderBottomLeftRadius:
                                            message.sender === "bot"
                                                ? "4px"
                                                : "16px",
                                        boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                                    }}
                                >
                                    <p
                                        style={{
                                            margin: 0,
                                            fontSize: "15px",
                                            lineHeight: "1.5",
                                            whiteSpace: "pre-wrap",
                                            wordWrap: "break-word",
                                        }}
                                    >
                                        {message.text}
                                    </p>
                                </div>

                                {message.sender === "user" && (
                                    <div
                                        style={{
                                            width: "32px",
                                            height: "32px",
                                            background: userIconColor,
                                            borderRadius: "50%",
                                            display: "flex",
                                            alignItems: "center",
                                            justifyContent: "center",
                                            flexShrink: 0,
                                            boxShadow:
                                                "0 2px 8px rgba(74, 144, 226, 0.2)",
                                        }}
                                    >
                                        <span
                                            style={{
                                                fontSize: "14px",
                                                filter: "brightness(10)",
                                            }}
                                        >
                                            👤
                                        </span>
                                    </div>
                                )}
                            </div>
                        ))}

                        {isTyping && (
                            <div
                                style={{
                                    display: "flex",
                                    alignItems: "flex-end",
                                    gap: "8px",
                                }}
                            >
                                <div
                                    style={{
                                        width: "32px",
                                        height: "32px",
                                        background: botIconColor,
                                        borderRadius: "12px",
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        flexShrink: 0,
                                        color: buttonColor,
                                    }}
                                >
                                    <MessageCircleIcon size={18} />
                                </div>
                                <div
                                    style={{
                                        backgroundColor: "#F3F4F6",
                                        borderRadius: "16px",
                                        borderBottomLeftRadius: "4px",
                                        padding: "12px 16px",
                                    }}
                                >
                                    <div
                                        style={{ display: "flex", gap: "4px" }}
                                    >
                                        <div
                                            style={{
                                                width: "8px",
                                                height: "8px",
                                                backgroundColor: "#9CA3AF",
                                                borderRadius: "50%",
                                                animation: "bounce 1s infinite",
                                            }}
                                        ></div>
                                        <div
                                            style={{
                                                width: "8px",
                                                height: "8px",
                                                backgroundColor: "#9CA3AF",
                                                borderRadius: "50%",
                                                animation:
                                                    "bounce 1s infinite 0.1s",
                                            }}
                                        ></div>
                                        <div
                                            style={{
                                                width: "8px",
                                                height: "8px",
                                                backgroundColor: "#9CA3AF",
                                                borderRadius: "50%",
                                                animation:
                                                    "bounce 1s infinite 0.2s",
                                            }}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>

                    {/* Quick Replies */}
                    {!hasUserMessage && (
                        <div style={{ padding: "0 16px 12px" }}>
                            <div
                                style={{
                                    display: "flex",
                                    flexWrap: "wrap",
                                    gap: "8px",
                                }}
                            >
                                {quickReplies.map((reply, index) => (
                                    <button
                                        key={index}
                                        onClick={() => handleQuickReply(reply)}
                                        style={{
                                            padding: "8px 16px",
                                            backgroundColor: "#F9FAFB",
                                            border: "1px solid #E5E7EB",
                                            borderRadius: "20px",
                                            fontSize: "14px",
                                            color: "#374151",
                                            cursor: "pointer",
                                            transition: "all 0.2s",
                                        }}
                                        onMouseEnter={(e) => {
                                            e.currentTarget.style.backgroundColor =
                                                botIconColor
                                            e.currentTarget.style.borderColor =
                                                buttonColor
                                        }}
                                        onMouseLeave={(e) => {
                                            e.currentTarget.style.backgroundColor =
                                                "#F9FAFB"
                                            e.currentTarget.style.borderColor =
                                                "#E5E7EB"
                                        }}
                                    >
                                        {reply}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                    {/* Input Area */}
                    <div
                        style={{
                            padding: "16px",
                            borderTop: "1px solid #E5E7EB",
                            backgroundColor: "white",
                        }}
                    >
                        <div
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "8px",
                                backgroundColor: "white",
                                borderRadius: "24px",
                                padding: "12px 16px",
                                border: "1px solid #D1D5DB",
                            }}
                        >
                            <input
                                type="text"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyPress={handleKeyPress}
                                placeholder="Ask anything..."
                                style={{
                                    flex: 1,
                                    border: "none",
                                    outline: "none",
                                    fontSize: "14px",
                                    color: "#374151",
                                    backgroundColor: "transparent",
                                }}
                            />
                            <button
                                style={{
                                    background: "transparent",
                                    border: "none",
                                    padding: "4px",
                                    cursor: "pointer",
                                    borderRadius: "8px",
                                    display: "flex",
                                    alignItems: "center",
                                    color: "#9CA3AF",
                                }}
                            >
                                <PaperclipIcon />
                            </button>
                            <button
                                onClick={handleSendMessage}
                                disabled={!inputValue.trim()}
                                style={{
                                    backgroundColor: inputValue.trim()
                                        ? sendButtonColor
                                        : "#D1D5DB",
                                    border: "none",
                                    borderRadius: "50%",
                                    width: "32px",
                                    height: "32px",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    cursor: inputValue.trim()
                                        ? "pointer"
                                        : "not-allowed",
                                    color: "white",
                                    transition: "all 0.2s",
                                }}
                            >
                                <SendIcon />
                            </button>
                        </div>
                    </div>
                </div>
            )}

            <style>{`
                @keyframes bounce {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-8px); }
                }
            `}</style>
        </>
    )
}
