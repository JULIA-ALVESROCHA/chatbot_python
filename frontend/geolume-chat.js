/*!
 * GeoLUME Chat Widget - OBG (Olimpiada Brasileira de Geografia)
 * Widget de chat autocontido para WordPress (sem dependencias, sem build).
 *
 * Instalacao via WPCode: cole este arquivo inteiro num snippet do tipo JavaScript.
 * Instalacao via arquivo: carregue geolume-chat.js com uma tag script no rodape.
 * Configuracao opcional: defina window.GeoLumeConfig antes deste codigo.
 */
(function () {
  "use strict";
  if (window.__geolumeLoaded) return;
  window.__geolumeLoaded = true;

  /* ---------------- Config ---------------- */
  var cfg = Object.assign(
    {
      apiUrl: "https://chatbotpython-production.up.railway.app/api/v1/chat",
      botName: "GeoLUME",
      subtitle: "Assistente virtual da OBG",
      greeting: "Olá! O que você quer saber sobre a OBG hoje?",
      language: "pt",
      quickReplies: [
        "Qual é o prazo de inscrição?",
        "Como funciona a avaliação?",
        "Posso participar individualmente?",
        "Onde encontro o regulamento?",
      ],
      disclaimer:
        "O GeoLUME é uma IA e pode cometer erros. Confira as respostas nos documentos oficiais.",
      position: "right", // "right" | "left"
      zIndex: 999999,
    },
    window.GeoLumeConfig || {}
  );

  /* ---------------- Estilos ---------------- */
  var css = "\
.gl-widget,.gl-widget *{box-sizing:border-box;margin:0;padding:0}\
.gl-widget{--gl-blue:#2E7FD6;--gl-teal:#2BB3C0;--gl-green:#3BC98E;--gl-yellow:#FFD166;--gl-text:#1F3140;--gl-muted:#75909E;--gl-font:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;font-family:var(--gl-font)}\
.gl-orb{background:radial-gradient(circle at 32% 28%,var(--gl-yellow) 0%,var(--gl-green) 48%,var(--gl-teal) 100%);border-radius:50%}\
.gl-launch{position:fixed;bottom:22px;width:62px;height:62px;border:none;border-radius:50%;background:var(--gl-green);color:#fff;display:flex;align-items:center;justify-content:center;cursor:pointer;box-shadow:0 8px 28px rgba(43,179,192,.45),0 2px 8px rgba(59,201,142,.3);transition:transform .2s ease,box-shadow .2s ease}\
.gl-launch:hover{transform:translateY(-2px) scale(1.06);box-shadow:0 12px 34px rgba(43,179,192,.55)}\
.gl-launch:focus-visible{outline:3px solid var(--gl-blue);outline-offset:3px}\
.gl-panel{position:fixed;bottom:22px;width:392px;max-width:calc(100vw - 32px);height:660px;max-height:calc(100dvh - 44px);border-radius:28px;overflow:hidden;display:flex;flex-direction:column;background:#FAFDFF;background-image:radial-gradient(ellipse 80% 55% at 78% 8%,rgba(174,236,208,.55),transparent 60%),radial-gradient(ellipse 70% 50% at 12% 30%,rgba(176,218,255,.5),transparent 65%),radial-gradient(ellipse 75% 45% at 85% 88%,rgba(255,241,188,.45),transparent 60%),radial-gradient(ellipse 60% 50% at 20% 92%,rgba(188,240,224,.4),transparent 60%);box-shadow:0 24px 70px rgba(28,84,126,.28);opacity:0;transform:translateY(16px) scale(.96);pointer-events:none;transition:opacity .24s ease,transform .24s ease}\
.gl-panel.gl-open{opacity:1;transform:none;pointer-events:auto}\
.gl-side-right{right:22px}.gl-side-left{left:22px}\
.gl-top{display:flex;align-items:center;justify-content:space-between;padding:16px 16px 4px;flex-shrink:0}\
.gl-iconbtn{width:36px;height:36px;border:none;border-radius:50%;background:rgba(255,255,255,.75);backdrop-filter:blur(8px);color:var(--gl-text);cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 10px rgba(28,84,126,.12);transition:background .15s,transform .15s}\
.gl-iconbtn:hover{background:#fff;transform:scale(1.06)}\
.gl-iconbtn:focus-visible{outline:2px solid var(--gl-blue);outline-offset:2px}\
.gl-topleft{display:flex;align-items:center;gap:9px}\
.gl-brand{display:flex;align-items:center;gap:9px;color:var(--gl-text)}\
.gl-brand .gl-orb{width:26px;height:26px;flex-shrink:0}\
.gl-brand-txt{display:flex;flex-direction:column;line-height:1.2}\
.gl-brand-name{font-size:14px;font-weight:700;letter-spacing:.2px}\
.gl-brand-sub{font-size:11px;font-weight:500;color:var(--gl-muted)}\
.gl-hero{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:34px 32px 22px;flex-shrink:0;transition:opacity .3s,max-height .35s,padding .35s;max-height:320px;overflow:hidden}\
.gl-hero .gl-orb{width:96px;height:96px;box-shadow:0 14px 44px rgba(59,201,142,.35),0 4px 60px rgba(43,179,192,.3);animation:gl-float 5s ease-in-out infinite}\
@keyframes gl-float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}\
.gl-hero h2{margin-top:26px;font-size:19px;font-weight:600;color:var(--gl-text);line-height:1.4;max-width:260px}\
.gl-hero.gl-hide{opacity:0;max-height:0;padding-top:0;padding-bottom:0;pointer-events:none}\
.gl-msgs{flex:1;overflow-y:auto;padding:10px 16px 14px;display:flex;flex-direction:column;gap:12px;scroll-behavior:smooth}\
.gl-msgs::-webkit-scrollbar{width:5px}.gl-msgs::-webkit-scrollbar-thumb{background:rgba(110,155,180,.3);border-radius:3px}\
.gl-row{display:flex;align-items:flex-end;gap:8px;animation:gl-in .25s ease both}\
.gl-row.gl-user{justify-content:flex-end}\
@keyframes gl-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}\
.gl-bot-ic{width:28px;height:28px;flex-shrink:0;box-shadow:0 3px 10px rgba(59,201,142,.3)}\
.gl-bubble{max-width:80%;padding:12px 15px;border-radius:20px;font-size:14.5px;line-height:1.55;white-space:pre-wrap;overflow-wrap:break-word;word-break:break-word}\
.gl-row.gl-bot .gl-bubble{background:rgba(255,255,255,.82);backdrop-filter:blur(10px);color:var(--gl-text);border-bottom-left-radius:6px;box-shadow:0 3px 14px rgba(28,84,126,.1)}\
.gl-row.gl-user .gl-bubble{background:linear-gradient(135deg,var(--gl-teal),var(--gl-blue));color:#fff;border-bottom-right-radius:6px;box-shadow:0 4px 16px rgba(46,127,214,.35)}\
.gl-time{display:block;font-size:10.5px;margin-top:5px;opacity:.55}\
.gl-sources{margin-top:9px;padding-top:8px;border-top:1px dashed rgba(110,155,180,.35);font-size:12px;color:var(--gl-muted)}\
.gl-sources strong{display:block;font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;color:var(--gl-blue);margin-bottom:4px}\
.gl-src{display:flex;gap:6px;align-items:baseline;margin-top:3px}\
.gl-src-badge{background:rgba(43,179,192,.12);color:var(--gl-blue);border-radius:8px;padding:1px 7px;font-size:11px;font-weight:600;white-space:normal;overflow-wrap:break-word;text-decoration:none;line-height:1.4}\
.gl-typing{display:flex;gap:5px;padding:14px 16px;background:rgba(255,255,255,.82);backdrop-filter:blur(10px);border-radius:20px;border-bottom-left-radius:6px;width:max-content;box-shadow:0 3px 14px rgba(28,84,126,.1)}\
.gl-dot{width:7px;height:7px;border-radius:50%;background:linear-gradient(135deg,var(--gl-green),var(--gl-teal));animation:gl-b 1.1s infinite}\
.gl-dot:nth-child(2){animation-delay:.15s}.gl-dot:nth-child(3){animation-delay:.3s}\
@keyframes gl-b{0%,100%{transform:translateY(0);opacity:.5}50%{transform:translateY(-6px);opacity:1}}\
.gl-quick{padding:4px 16px 10px;display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end;flex-shrink:0}\
.gl-chip{padding:10px 14px;background:rgba(255,255,255,.9);backdrop-filter:blur(8px);border:none;border-radius:14px;font-size:12px;line-height:1.35;color:var(--gl-text);cursor:pointer;font-family:inherit;text-align:left;max-width:170px;box-shadow:0 3px 12px rgba(28,84,126,.1);transition:transform .15s,box-shadow .15s}\
.gl-chip:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(46,127,214,.2)}\
.gl-chip:focus-visible{outline:2px solid var(--gl-blue);outline-offset:2px}\
.gl-inputbar{padding:0 14px 10px;flex-shrink:0}\
.gl-inputwrap{display:flex;align-items:flex-end;gap:8px;background:rgba(255,255,255,.92);backdrop-filter:blur(12px);border-radius:24px;padding:9px 9px 9px 18px;box-shadow:0 6px 22px rgba(28,84,126,.16);transition:box-shadow .15s}\
.gl-inputwrap:focus-within{box-shadow:0 6px 22px rgba(46,127,214,.3),0 0 0 2px rgba(43,179,192,.35)}\
.gl-input{flex:1;border:none;outline:none;background:transparent;font-size:14.5px;font-family:inherit;color:var(--gl-text);resize:none;max-height:96px;line-height:1.45;padding:5px 0}\
.gl-input::placeholder{color:var(--gl-muted)}\
.gl-send{width:38px;height:38px;border:none;border-radius:50%;background:linear-gradient(135deg,var(--gl-teal),var(--gl-blue));color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 4px 14px rgba(46,127,214,.4);transition:transform .15s,opacity .15s}\
.gl-send:hover:not(:disabled){transform:scale(1.1)}\
.gl-send:disabled{opacity:.4;cursor:not-allowed;box-shadow:none}\
.gl-send:focus-visible{outline:2px solid var(--gl-green);outline-offset:2px}\
.gl-disclaimer{text-align:center;font-size:10.5px;line-height:1.45;color:var(--gl-muted);padding:0 22px 12px;flex-shrink:0}\
.gl-error .gl-bubble{background:rgba(255,240,243,.95);color:#A03050}\
@media(max-width:480px){.gl-panel{right:0!important;left:0!important;bottom:0;width:100vw;max-width:100vw;height:100dvh;max-height:100dvh;border-radius:0}.gl-launch{bottom:16px}.gl-hero .gl-orb{width:84px;height:84px}.gl-chip{max-width:none;flex:1 1 45%}}\
@media(prefers-reduced-motion:reduce){.gl-panel,.gl-row,.gl-launch,.gl-send,.gl-chip{transition:none;animation:none}.gl-dot,.gl-hero .gl-orb{animation:none}}\
";

  var style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  /* ---------------- Ícones ---------------- */
  function svg(inner, size) {
    return (
      '<svg width="' + (size || 20) + '" height="' + (size || 20) +
      '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      inner + "</svg>"
    );
  }
  var icX = '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>';
  var icUp = '<line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>';
  var icBack = '<polyline points="15 18 9 12 15 6"/>';
  var icMsg = '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>';

  /* ---------------- Estado ---------------- */
  var sessionId =
    "session_" + Date.now() + "_" + Math.random().toString(36).slice(2, 11);
  var isTyping = false;
  var hasUserMessage = false;
  var side = cfg.position === "left" ? "gl-side-left" : "gl-side-right";

  /* ---------------- DOM ---------------- */
  var launch = document.createElement("button");
  launch.className = "gl-widget gl-launch " + side;
  launch.innerHTML = svg(icMsg, 28);
  launch.style.zIndex = cfg.zIndex;
  launch.setAttribute("aria-label", "Abrir chat com " + cfg.botName);

  var panel = document.createElement("div");
  panel.className = "gl-widget gl-panel " + side;
  panel.style.zIndex = cfg.zIndex;
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Chat " + cfg.botName);
  panel.innerHTML =
    '<div class="gl-top">' +
      '<div class="gl-topleft">' +
        '<button class="gl-iconbtn gl-back" aria-label="Voltar ao início" hidden>' + svg(icBack, 18) + "</button>" +
        '<div class="gl-brand"><span class="gl-orb"></span>' +
          '<span class="gl-brand-txt"><span class="gl-brand-name"></span><span class="gl-brand-sub"></span></span>' +
        "</div>" +
      "</div>" +
      '<button class="gl-iconbtn gl-close" aria-label="Fechar chat">' + svg(icX, 18) + "</button>" +
    "</div>" +
    '<div class="gl-hero"><div class="gl-orb"></div><h2></h2></div>' +
    '<div class="gl-msgs" aria-live="polite"></div>' +
    '<div class="gl-quick"></div>' +
    '<div class="gl-inputbar">' +
      '<div class="gl-inputwrap">' +
        '<textarea class="gl-input" rows="1" placeholder="Comece a perguntar..." aria-label="Mensagem"></textarea>' +
        '<button class="gl-send" disabled aria-label="Enviar mensagem">' + svg(icUp, 17) + "</button>" +
      "</div>" +
    "</div>" +
    '<div class="gl-disclaimer"></div>';

  panel.querySelector(".gl-brand-name").textContent = cfg.botName;
  panel.querySelector(".gl-brand-sub").textContent = cfg.subtitle;
  panel.querySelector(".gl-hero h2").textContent = cfg.greeting;
  panel.querySelector(".gl-disclaimer").textContent = cfg.disclaimer;

  document.body.appendChild(launch);
  document.body.appendChild(panel);

  var heroEl = panel.querySelector(".gl-hero");
  var backEl = panel.querySelector(".gl-back");
  var msgsEl = panel.querySelector(".gl-msgs");
  var quickEl = panel.querySelector(".gl-quick");
  var inputEl = panel.querySelector(".gl-input");
  var sendEl = panel.querySelector(".gl-send");

  /* ---------------- Helpers ---------------- */
  function timeNow() {
    return new Date().toLocaleTimeString("pt-BR", {
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function scrollDown() {
    msgsEl.scrollTop = msgsEl.scrollHeight;
  }

  function addMessage(text, sender, sources, isError) {
    var row = document.createElement("div");
    row.className =
      "gl-row " + (sender === "user" ? "gl-user" : "gl-bot") +
      (isError ? " gl-error" : "");

    if (sender === "bot") {
      var ic = document.createElement("div");
      ic.className = "gl-bot-ic gl-orb";
      row.appendChild(ic);
    }

    var bubble = document.createElement("div");
    bubble.className = "gl-bubble";
    var p = document.createElement("span");
    p.textContent = text; // textContent = seguro contra XSS
    bubble.appendChild(p);

    if (sources && sources.length) {
      var box = document.createElement("div");
      box.className = "gl-sources";
      var t = document.createElement("strong");
      t.textContent = "Fontes";
      box.appendChild(t);
      sources.slice(0, 4).forEach(function (s) {
        var line = document.createElement("div");
        line.className = "gl-src";
        var label;
        if (s && s.citation) {
          label = s.citation;
        } else {
          var name =
            (s && (s.title || s.source_id || s.source || s.document)) ||
            "Documento OBG";
          label = name + (s && s.page != null ? " · p. " + s.page : "");
        }
        var badge;
        if (s && s.url) {
          badge = document.createElement("a");
          badge.href = s.url;
          badge.target = "_blank";
          badge.rel = "noopener noreferrer";
        } else {
          badge = document.createElement("span");
        }
        badge.className = "gl-src-badge";
        badge.textContent = label;
        line.appendChild(badge);
        var ex = s && (s.excerpt || s.snippet || s.section);
        if (ex) {
          var exEl = document.createElement("span");
          exEl.textContent = String(ex).slice(0, 120);
          line.appendChild(exEl);
        }
        box.appendChild(line);
      });
      bubble.appendChild(box);
    }

    var time = document.createElement("span");
    time.className = "gl-time";
    time.textContent = timeNow();
    bubble.appendChild(time);

    row.appendChild(bubble);
    msgsEl.appendChild(row);
    scrollDown();
  }

  var typingEl = null;
  function showTyping(on) {
    isTyping = on;
    sendEl.disabled = on || !inputEl.value.trim();
    if (on) {
      typingEl = document.createElement("div");
      typingEl.className = "gl-row gl-bot";
      typingEl.innerHTML =
        '<div class="gl-bot-ic gl-orb"></div>' +
        '<div class="gl-typing"><span class="gl-dot"></span><span class="gl-dot"></span><span class="gl-dot"></span></div>';
      msgsEl.appendChild(typingEl);
      scrollDown();
    } else if (typingEl) {
      typingEl.remove();
      typingEl = null;
    }
  }

  function showHome() {
    heroEl.classList.remove("gl-hide");
    msgsEl.style.display = "none";
    backEl.hidden = true;
    renderQuickReplies();
  }

  function showChat() {
    heroEl.classList.add("gl-hide");
    msgsEl.style.display = "flex";
    quickEl.style.display = "none";
    backEl.hidden = false;
  }

  function renderQuickReplies() {
    quickEl.innerHTML = "";
    quickEl.style.display = "flex";
    if (!cfg.quickReplies || !cfg.quickReplies.length) {
      quickEl.style.display = "none";
      return;
    }
    cfg.quickReplies.forEach(function (q) {
      var b = document.createElement("button");
      b.className = "gl-chip";
      b.type = "button";
      b.textContent = q;
      b.addEventListener("click", function () {
        sendMessage(q);
      });
      quickEl.appendChild(b);
    });
  }

  /* ---------------- API ---------------- */
  function sendMessage(text) {
    text = (text || "").trim();
    if (!text || isTyping) return;

    hasUserMessage = true;
    showChat();

    addMessage(text, "user");
    inputEl.value = "";
    inputEl.style.height = "auto";
    sendEl.disabled = true;
    showTyping(true);

    var controller =
      typeof AbortController !== "undefined" ? new AbortController() : null;
    var timeout = controller
      ? setTimeout(function () {
          controller.abort();
        }, 60000)
      : null;

    fetch(cfg.apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: text,
        session_id: sessionId,
        language: cfg.language,
      }),
      signal: controller ? controller.signal : undefined,
    })
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then(function (data) {
        showTyping(false);
        var answer = data.answer || "Não encontrei essa informação nos documentos oficiais.";
        // Remove bloco de citações duplicado no texto (as fontes aparecem nos badges)
        answer = answer
          .replace(/\n*\s*(você pode encontrar( mais)? em|encontre mais em|consulte|fontes?|referências?)\s*:[\s\S]*$/i, "")
          .trim() || answer;
        addMessage(
          answer,
          "bot",
          Array.isArray(data.sources) ? data.sources : []
        );
      })
      .catch(function () {
        showTyping(false);
        addMessage(
          "Não foi possível conectar ao servidor agora. Tente novamente em instantes.",
          "bot",
          null,
          true
        );
      })
      .finally(function () {
        if (timeout) clearTimeout(timeout);
        inputEl.focus();
      });
  }

  /* ---------------- Eventos ---------------- */
  function openChat() {
    panel.classList.add("gl-open");
    launch.style.display = "none";
    setTimeout(function () {
      inputEl.focus();
    }, 120);
  }
  function closeChat() {
    panel.classList.remove("gl-open");
    launch.style.display = "flex";
    launch.focus();
  }

  launch.addEventListener("click", openChat);
  panel.querySelector(".gl-close").addEventListener("click", closeChat);
  backEl.addEventListener("click", showHome);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && panel.classList.contains("gl-open")) closeChat();
  });

  inputEl.addEventListener("input", function () {
    sendEl.disabled = isTyping || !inputEl.value.trim();
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 96) + "px";
  });
  inputEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputEl.value);
    }
  });
  sendEl.addEventListener("click", function () {
    sendMessage(inputEl.value);
  });

  /* ---------------- Init ---------------- */
  showHome();
})();