"""
notifier.py — Sistema de notificaciones AquaCare.

Escalado progresivo inspirado en la banca:
  VIGILANCIA → Telegram a operaciones AMAEM
  ALTO       → Telegram + llamada Vapi (IA conversacional en español)
  CRÍTICO    → Telegram + llamada Vapi → si no contesta → contacto emergencia

Tecnologías:
  - Telegram Bot API   (gratis, via requests)
  - Vapi.ai            (IA de voz conversacional, trial gratis)
  - Twilio             (fallback TTS si Vapi no configurado)

Configuración (.env):
  TELEGRAM_BOT_TOKEN=xxx
  TELEGRAM_CHAT_ID=xxx
  VAPI_API_KEY=xxx
  VAPI_PHONE_NUMBER_ID=xxx
  CONTACT_PHONE_NUMBER=+34XXXXXXXXX     # Titular / operador AMAEM
  EMERGENCY_CONTACT_PHONE=+34XXXXXXXXX  # Familiar o servicios sociales
  TWILIO_ACCOUNT_SID=xxx                # Fallback
  TWILIO_AUTH_TOKEN=xxx
  TWILIO_FROM_NUMBER=+1XXXXXXXXXX
  AQUAGUARD_DEMO_MODE=true              # En demo: no espera respuesta
  ESCALATION_WAIT_SECONDS=300           # 5 min para demo, bajar a 10

Uso:
  from notifier import send_welfare_notifications
  send_welfare_notifications(welfare_alerts, notify_telegram=True, notify_voice=True)
"""

import io
import os
import sys
import time
import logging
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv

# Force UTF-8 output on Windows (emojis en mensajes Telegram)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuración desde .env ──────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

VAPI_API_KEY = os.getenv("VAPI_API_KEY", "")
VAPI_PHONE_ID = os.getenv("VAPI_PHONE_NUMBER_ID", "")

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER", "")

CONTACT_PHONE = os.getenv("CONTACT_PHONE_NUMBER", "")
EMERGENCY_PHONE = os.getenv("EMERGENCY_CONTACT_PHONE", "")

DEMO_MODE = os.getenv("AQUAGUARD_DEMO_MODE", "false").lower() == "true"
ESCALATION_WAIT = int(os.getenv("ESCALATION_WAIT_SECONDS", "300"))


# ═══════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════

def _format_telegram_message(alert: pd.Series) -> str:
    """Formatea el mensaje Telegram con HTML. Visual, impactante, accionable."""
    nivel = alert.get("nivel", "VIGILANCIA")
    barrio = alert.get("barrio", "N/A")
    drop_pct = alert.get("drop_pct", 0)
    vuln = alert.get("elderly_vulnerability", 0)
    streak = alert.get("consecutive_decline_months", 0)
    confidence = alert.get("confidence", 0)
    mensaje = alert.get("mensaje", "")
    other_models = alert.get("other_models_confirming", 0)

    # Emojis según nivel
    nivel_emoji = {"CRITICO": "🚨", "ALTO": "🔶", "VIGILANCIA": "🟡"}.get(nivel, "ℹ️")
    nivel_color = {"CRITICO": "🔴", "ALTO": "🟠", "VIGILANCIA": "🟡"}.get(nivel, "⚪")

    # Contexto demográfico (si disponible)
    pct_elderly = alert.get("pct_elderly_65plus", None)
    pct_alone = alert.get("pct_elderly_alone", None)

    demo_line = ""
    if pct_elderly and pct_elderly > 0:
        demo_line = f"\n👴 <b>Población mayor:</b> {pct_elderly:.1f}% (≥65 años)"
    if pct_alone and pct_alone > 0:
        demo_line += f"\n🏠 <b>Viven solos:</b> {pct_alone:.1f}% de los mayores"

    models_line = ""
    if other_models and other_models > 0:
        models_line = f"\n🤖 <b>Modelos que confirman:</b> {other_models} de 6"

    action = {
        "CRITICO":    "🔴 <b>VERIFICACIÓN PRESENCIAL INMEDIATA</b>",
        "ALTO":       "🟠 Contactar titular + revisar contadores individuales",
        "VIGILANCIA": "🟡 Monitoreo estrecho — revisar próximo mes",
    }.get(nivel, "Revisión estándar")

    msg = (
        f"{nivel_emoji} <b>ALERTA AQUACARE — AquaGuard AI</b>\n"
        f"\n"
        f"📍 <b>Barrio:</b> {barrio}\n"
        f"{nivel_color} <b>Nivel:</b> {nivel}\n"
        f"📉 <b>Caída de consumo:</b> -{drop_pct:.1f}%\n"
        f"📅 <b>Meses consecutivos:</b> {streak}"
        f"{demo_line}"
        f"{models_line}\n"
        f"\n"
        f"⚠️ <i>{mensaje[:200] if mensaje else 'Posible fuga silenciosa o situación de vulnerabilidad.'}</i>\n"
        f"\n"
        f"→ {action}\n"
        f"\n"
        f"<i>AquaGuard AI · AMAEM · Confianza: {confidence:.0%}</i>"
    )
    return msg


def send_telegram_alert(alert: pd.Series,
                        token: str = TELEGRAM_TOKEN,
                        chat_id: str = TELEGRAM_CHAT_ID) -> bool:
    """
    Envía un mensaje Telegram para una alerta AquaCare.
    Retorna True si se envió correctamente.
    """
    if not token or not chat_id:
        logger.warning("Telegram no configurado (TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID vacíos)")
        return False

    message = _format_telegram_message(alert)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            nivel = alert.get("nivel", "?")
            barrio = alert.get("barrio", "?")
            print(f"    📱 Telegram enviado: [{nivel}] {barrio}")
            return True
        else:
            logger.error(f"Telegram error {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.RequestException as e:
        logger.error(f"Telegram request failed: {e}")
        return False


def send_batch_telegram(welfare_alerts: pd.DataFrame,
                        levels: list = None,
                        token: str = TELEGRAM_TOKEN,
                        chat_id: str = TELEGRAM_CHAT_ID) -> int:
    """
    Envía Telegram para todas las alertas de los niveles indicados.
    levels: ["CRITICO", "ALTO", "VIGILANCIA"] — por defecto CRITICO + ALTO
    Retorna número de mensajes enviados.
    """
    if levels is None:
        levels = ["CRITICO", "ALTO"]

    if welfare_alerts is None or len(welfare_alerts) == 0:
        return 0

    subset = welfare_alerts[welfare_alerts["nivel"].isin(levels)]
    sent = 0
    for _, alert in subset.iterrows():
        if send_telegram_alert(alert, token, chat_id):
            sent += 1
        time.sleep(0.3)  # rate limit Telegram

    return sent


# ═══════════════════════════════════════════════════════════════════
# VAPI — Llamada de voz con IA conversacional en español
# ═══════════════════════════════════════════════════════════════════

def _build_vapi_assistant(alert: pd.Series) -> dict:
    """
    Construye la configuración del asistente Vapi para una alerta específica.
    El agente puede CONVERSAR — no es solo TTS pregrabado.
    """
    barrio = alert.get("barrio", "su barrio")
    nivel = alert.get("nivel", "ALTO")

    system_content = (
        f"Eres un asistente del sistema AquaGuard AI de AMAEM, la empresa de agua de Alicante. "
        f"Llamas al titular del suministro de agua del barrio {barrio} porque hemos detectado "
        f"una anomalía {nivel.lower()} en el consumo de agua. "
        f"Tu objetivo es verificar que la persona está bien y avisarle del problema. "
        f"Sé breve (máximo 2 minutos), profesional y empático. "
        f"Si dice que está bien, agradece y despídete indicando que AMAEM dará seguimiento. "
        f"Si dice que necesita ayuda o hay una emergencia, di que conectas con servicios de emergencias "
        f"y que alguien irá a su domicilio. "
        f"Habla siempre en español de España."
    )

    first_message = (
        f"Buenos días, le llama el sistema AquaGuard AI de AMAEM. "
        f"Hemos detectado una posible anomalía en el suministro de agua de {barrio}. "
        f"¿Está usted bien? ¿Ha notado algo inusual con el agua en su domicilio?"
    )

    return {
        "transcriber": {
            "provider": "deepgram",
            "language": "es",
            "model": "nova-2",
        },
        "model": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "messages": [{"role": "system", "content": system_content}],
            "temperature": 0.3,
        },
        "voice": {
            "provider": "azure",
            "voiceId": "es-ES-ElviraNeural",
        },
        "firstMessage": first_message,
        "endCallPhrases": ["adiós", "hasta luego", "gracias adiós", "de nada"],
        "silenceTimeoutSeconds": 30,
        "maxDurationSeconds": 120,
        "backgroundSound": "off",
    }


def make_vapi_call(phone: str, alert: pd.Series,
                   api_key: str = VAPI_API_KEY,
                   phone_id: str = VAPI_PHONE_ID) -> Optional[str]:
    """
    Inicia una llamada de voz via Vapi.ai.
    Retorna call_id si la llamada se inició correctamente, None si falló.
    """
    if not api_key or not phone_id:
        logger.warning("Vapi no configurado (VAPI_API_KEY o VAPI_PHONE_NUMBER_ID vacíos)")
        return None

    if not phone:
        logger.warning("Número de teléfono de contacto no configurado")
        return None

    barrio = alert.get("barrio", "N/A")
    nivel = alert.get("nivel", "ALTO")

    payload = {
        "phoneNumberId": phone_id,
        "customer": {
            "number": phone,
            "name": f"Titular {barrio}",
        },
        "assistant": _build_vapi_assistant(alert),
        "metadata": {
            "barrio": barrio,
            "nivel": nivel,
            "source": "AquaGuard-AquaCare",
        },
    }

    try:
        resp = requests.post(
            "https://api.vapi.ai/call/phone",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        if resp.status_code in (200, 201):
            call_id = resp.json().get("id", "unknown")
            print(f"    📞 Llamada Vapi iniciada: [{nivel}] {barrio} → {phone} (id: {call_id[:8]}...)")
            return call_id
        else:
            logger.error(f"Vapi error {resp.status_code}: {resp.text[:300]}")
            return None
    except requests.RequestException as e:
        logger.error(f"Vapi request failed: {e}")
        return None


def check_vapi_call_status(call_id: str, api_key: str = VAPI_API_KEY) -> str:
    """
    Consulta el estado de una llamada Vapi.
    Retorna: "ringing", "in-progress", "ended", "no-answer", "failed"
    """
    if not api_key or not call_id:
        return "unknown"

    try:
        resp = requests.get(
            f"https://api.vapi.ai/call/{call_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "unknown")
            ended_reason = data.get("endedReason", "")
            if ended_reason in ("customer-did-not-answer", "no-answer"):
                return "no-answer"
            return status
        return "unknown"
    except requests.RequestException:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════
# TWILIO — Fallback TTS si Vapi no está configurado
# ═══════════════════════════════════════════════════════════════════

def make_twilio_call(phone: str, alert: pd.Series,
                     sid: str = TWILIO_SID,
                     token: str = TWILIO_TOKEN,
                     from_number: str = TWILIO_FROM) -> Optional[str]:
    """
    Llamada de voz TTS via Twilio (fallback cuando Vapi no está configurado).
    Retorna el call SID si se inició correctamente.
    """
    if not sid or not token or not from_number:
        logger.warning("Twilio no configurado")
        return None

    if not phone:
        logger.warning("Número de contacto no configurado")
        return None

    barrio = alert.get("barrio", "su barrio")
    nivel = alert.get("nivel", "ALTO")

    twiml = (
        f"<Response>"
        f"<Say language='es-ES' voice='Polly.Conchita'>"
        f"Buenos días, le llama el sistema Aqua Guard AI de A M A E M. "
        f"Hemos detectado una posible fuga de agua en el barrio {barrio}. "
        f"El nivel de alerta es {nivel}. "
        f"Por favor, compruebe su suministro de agua. "
        f"Si necesita ayuda, contacte con AMAEM o con el 112. "
        f"Muchas gracias y hasta luego."
        f"</Say>"
        f"</Response>"
    )

    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json"
    auth = (sid, token)

    try:
        resp = requests.post(
            url, auth=auth,
            data={"To": phone, "From": from_number, "Twiml": twiml},
            timeout=15,
        )
        if resp.status_code == 201:
            call_sid = resp.json().get("sid", "unknown")
            print(f"    📞 Llamada Twilio iniciada: [{nivel}] {barrio} → {phone} (sid: {call_sid[:8]}...)")
            return call_sid
        else:
            logger.error(f"Twilio error {resp.status_code}: {resp.text[:300]}")
            return None
    except requests.RequestException as e:
        logger.error(f"Twilio request failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# ESCALATION MANAGER
# ═══════════════════════════════════════════════════════════════════

def _make_voice_call(phone: str, alert: pd.Series) -> Optional[str]:
    """Intenta Vapi primero, Twilio como fallback."""
    if VAPI_API_KEY and VAPI_PHONE_ID:
        return make_vapi_call(phone, alert)
    elif TWILIO_SID and TWILIO_TOKEN:
        return make_twilio_call(phone, alert)
    else:
        nivel = alert.get("nivel", "?")
        barrio = alert.get("barrio", "?")
        print(f"    ⚠️  Sin proveedor de voz configurado (Vapi/Twilio). "
              f"Llamada a [{nivel}] {barrio} → {phone} simulada.")
        return "DEMO_SIMULATED"


def escalate_alert(alert: pd.Series,
                   notify_telegram: bool = True,
                   notify_voice: bool = True) -> dict:
    """
    Protocolo completo de escalado para una alerta AquaCare.

    Niveles:
      VIGILANCIA → Solo Telegram
      ALTO       → Telegram + llamada voz al titular
      CRÍTICO    → Telegram + llamada voz al titular
                   → Si no contesta en ESCALATION_WAIT segundos:
                     llamada al contacto de emergencia

    Retorna dict con el resultado de cada paso.
    """
    nivel = alert.get("nivel", "VIGILANCIA")
    barrio = alert.get("barrio", "N/A")
    result = {"nivel": nivel, "barrio": barrio, "steps": []}

    print(f"\n  🚨 ESCALANDO [{nivel}] {barrio}")

    # Paso 1: Telegram (siempre para ALTO y CRÍTICO)
    if notify_telegram and nivel in ("CRITICO", "ALTO", "VIGILANCIA"):
        ok = send_telegram_alert(alert)
        result["steps"].append({"action": "telegram", "success": ok})

    # Paso 2: Llamada de voz para ALTO y CRÍTICO
    if notify_voice and nivel in ("CRITICO", "ALTO") and CONTACT_PHONE:
        call_id = _make_voice_call(CONTACT_PHONE, alert)
        result["steps"].append({"action": "voice_call_primary", "call_id": call_id})

        # Paso 3: Escalar si CRÍTICO y no contesta (solo si hay call_id real)
        if nivel == "CRITICO" and call_id and call_id != "DEMO_SIMULATED":
            wait = 30 if DEMO_MODE else ESCALATION_WAIT
            print(f"    ⏳ Esperando {wait}s respuesta antes de escalar...")
            time.sleep(wait)

            status = check_vapi_call_status(call_id) if VAPI_API_KEY else "no-answer"
            result["steps"].append({"action": "call_status_check", "status": status})

            if status in ("no-answer", "failed", "unknown") and EMERGENCY_PHONE:
                print(f"    📞 Sin respuesta. Escalando a contacto de emergencia...")
                em_alert = alert.copy()
                em_alert["mensaje"] = (
                    f"ESCALADO: {barrio} sin respuesta. Contacto principal no disponible. "
                    f"Requiere intervención presencial."
                )
                em_call_id = _make_voice_call(EMERGENCY_PHONE, em_alert)
                result["steps"].append({
                    "action": "voice_call_emergency",
                    "call_id": em_call_id,
                })

        elif nivel == "CRITICO" and call_id == "DEMO_SIMULATED" and DEMO_MODE:
            print(f"    🎭 [DEMO MODE] Simulando escalado: sin respuesta → contacto emergencia")
            result["steps"].append({"action": "demo_escalation_simulated", "success": True})

    return result


# ═══════════════════════════════════════════════════════════════════
# API PÚBLICA
# ═══════════════════════════════════════════════════════════════════

def send_welfare_notifications(
    welfare_alerts: pd.DataFrame,
    notify_telegram: bool = True,
    notify_voice: bool = False,
) -> list:
    """
    Punto de entrada principal. Procesa todas las alertas AquaCare.

    Args:
        welfare_alerts: DataFrame de alertas de welfare_detector.run_welfare_detection()
        notify_telegram: Enviar notificaciones Telegram (default: True)
        notify_voice: Realizar llamadas de voz para ALTO/CRÍTICO (default: False)

    Returns:
        Lista de dicts con resultados de cada alerta procesada.
    """
    if welfare_alerts is None or len(welfare_alerts) == 0:
        return []

    if not notify_telegram and not notify_voice:
        return []

    # Verificar configuración mínima
    if notify_telegram and (not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID):
        print("  ⚠️  Telegram no configurado. Añade TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID al .env")
        notify_telegram = False

    if notify_voice and not CONTACT_PHONE:
        print("  ⚠️  Sin número de contacto (CONTACT_PHONE_NUMBER). Llamadas desactivadas.")
        notify_voice = False

    if not notify_telegram and not notify_voice:
        return []

    n_critico = (welfare_alerts["nivel"] == "CRITICO").sum()
    n_alto = (welfare_alerts["nivel"] == "ALTO").sum()
    n_vigil = (welfare_alerts["nivel"] == "VIGILANCIA").sum()

    print(f"\n  📡 SISTEMA DE NOTIFICACIONES AQUACARE")
    print(f"     Procesando: {n_critico} CRÍTICO, {n_alto} ALTO, {n_vigil} VIGILANCIA")
    if DEMO_MODE:
        print(f"     [MODO DEMO ACTIVO — escalado instantáneo]")

    results = []
    for _, alert in welfare_alerts.iterrows():
        res = escalate_alert(alert, notify_telegram=notify_telegram, notify_voice=notify_voice)
        results.append(res)

    n_sent = sum(1 for r in results if any(s.get("success") for s in r.get("steps", [])))
    print(f"\n  ✅ Notificaciones enviadas: {n_sent}/{len(results)}")

    return results


# ═══════════════════════════════════════════════════════════════════
# DEMO — Prueba manual del sistema
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  TEST — Sistema de Notificaciones AquaCare")
    print("=" * 60)

    # Crear alerta de prueba
    test_alert = pd.Series({
        "barrio": "17-CAROLINAS ALTAS",
        "nivel": "CRITICO",
        "drop_pct": 47.3,
        "elderly_vulnerability": 0.78,
        "consecutive_decline_months": 4,
        "confidence": 0.91,
        "pct_elderly_65plus": 35.0,
        "pct_elderly_alone": 42.8,
        "other_models_confirming": 3,
        "mensaje": "Fuga silenciosa detectada. Consumo ha caído un 47% en 4 meses consecutivos.",
    })

    print(f"\nAlerta de prueba: [{test_alert['nivel']}] {test_alert['barrio']}")
    print("\nMensaje Telegram formateado:")
    print("-" * 60)
    msg = _format_telegram_message(test_alert)
    # Mostrar sin HTML
    import re
    print(re.sub(r"<[^>]+>", "", msg))
    print("-" * 60)

    print("\nConfig detectada:")
    print(f"  Telegram: {'✅' if TELEGRAM_TOKEN else '❌ No configurado'}")
    print(f"  Vapi:     {'✅' if VAPI_API_KEY else '❌ No configurado'}")
    print(f"  Twilio:   {'✅' if TWILIO_SID else '❌ No configurado'}")
    print(f"  Teléfono: {'✅ ' + CONTACT_PHONE[:6] + '...' if CONTACT_PHONE else '❌ No configurado'}")
    print(f"  Demo mode: {'✅' if DEMO_MODE else '❌'}")

    if TELEGRAM_TOKEN:
        print("\nEnviando Telegram de prueba...")
        ok = send_telegram_alert(test_alert)
        print(f"  Resultado: {'✅ Enviado' if ok else '❌ Error'}")
    else:
        print("\nPara probar: configura TELEGRAM_BOT_TOKEN en .env")
        print("  1. Crea un bot en @BotFather")
        print("  2. Escribe al bot para obtener tu chat_id")
        print("  3. Añade al .env y ejecuta: python notifier.py")
