# frontend.py
# LawyerBot Gradio Frontend Interface

import gradio as gr
import uuid
import json
import logging
from backend import LawyerBot

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global LawyerBot instance
lawyerbot = None

def initialize_lawyerbot():
    global lawyerbot
    try:
        lawyerbot = LawyerBot()
        logger.info("LawyerBot initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LawyerBot: {str(e)}")
        return False

def process_legal_query(question: str, context: str, disclaimer_agreed: bool, session_state: dict):
    if not disclaimer_agreed:
        return "", "", "⚠️ Please acknowledge the legal disclaimer before proceeding.", session_state
    if not question.strip():
        return "", "", "📝 Please enter a legal question.", session_state
    if 'session_id' not in session_state:
        session_state['session_id'] = str(uuid.uuid4())

    try:
        response = lawyerbot.get_response(
            user_text=question.strip(),
            user_context=context.strip() if context else None,
            session_id=session_state['session_id']
        )
        if response['success']:
            structured_answer = response['structured_answer']
            raw_output = json.dumps(response['raw_llm_output'], indent=2)
            logger.info(f"Successfully processed query for session: {session_state['session_id']}")
            return structured_answer, raw_output, "", session_state
        else:
            error_msg = f"⚠️ {response['structured_answer']}"
            return "", "", error_msg, session_state
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return "", "", f"❌ System error: {str(e)}", session_state

def clear_form():
    return "", "", False, "", ""

# Custom CSS
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.disclaimer-box { background-color: #2d1b69 !important; border: 2px solid #ffd700 !important; border-radius: 10px !important; padding: 15px !important; margin: 10px 0 !important; }
.legal-response { background-color: #0f1419 !important; border: 1px solid #533bae !important; border-radius: 8px !important; padding: 20px !important; font-family: 'Georgia', serif !important; line-height: 1.6 !important; }
.error-message { background-color: #4a1a1a !important; border: 1px solid #ff6b6b !important; border-radius: 5px !important; color: #ff6b6b !important; padding: 10px !important; }
.header-text { text-align: center; color: #ffd700 !important; font-weight: bold; margin-bottom: 20px; }
.primary-button { background: linear-gradient(45deg, #533bae, #ffd700) !important; border: none !important; color: white !important; font-weight: bold !important; }
.secondary-button { background-color: #2d1b69 !important; border: 1px solid #533bae !important; color: white !important; }
.input-field { background-color: #1a1a2e !important; border: 1px solid #533bae !important; color: white !important; }
.checkbox-label { color: #ffd700 !important; font-weight: bold !important; }
"""

def create_interface():
    disclaimer_text = """
⚖️ **IMPORTANT LEGAL DISCLAIMER** ⚖️
This AI assistant provides **GENERAL LEGAL INFORMATION ONLY** and is **NOT** a substitute for professional legal advice from a licensed attorney.
By checking the box below, you acknowledge that you understand these limitations.
"""

    with gr.Blocks(css=custom_css, title="LawyerBot - AI Legal Assistant", theme=gr.themes.Glass()) as interface:
        session_state = gr.State(value={})
        gr.HTML("""<div class="header-text"><h1>⚖️ LawyerBot - AI Legal Information Assistant</h1></div>""")

        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML(f"<div class='disclaimer-box'>{disclaimer_text.replace(chr(10), '<br>')}</div>")
                disclaimer_checkbox = gr.Checkbox(
                    label="✓ I understand this is general legal information only",
                    value=False,
                    elem_classes=["checkbox-label"]
                )
                question_input = gr.Textbox(
                    label="📝 Legal Question",
                    placeholder="Enter your legal question here",
                    lines=4,
                    elem_classes=["input-field"]
                )
                context_input = gr.Textbox(
                    label="📋 Additional Context (Optional)",
                    placeholder="Optional context",
                    lines=3,
                    elem_classes=["input-field"]
                )
                with gr.Row():
                    submit_btn = gr.Button("🔍 Get Legal Information", variant="primary", elem_classes=["primary-button"])
                    clear_btn = gr.Button("🗑️ Clear Form", variant="secondary", elem_classes=["secondary-button"])

            with gr.Column(scale=3):
                error_output = gr.HTML(label="Status")
                response_output = gr.Textbox(label="⚖️ Legal Information Response", lines=20, interactive=False, elem_classes=["legal-response"])
                with gr.Accordion("🔧 Debug Information (Advanced)", open=False):
                    debug_output = gr.Textbox(label="Raw LLM Output", lines=10, interactive=False)

        # Persistent disclaimer
        gr.HTML("<div style='text-align:center;color:#ffd700;font-weight:bold;margin-top:20px;'>⚖️ General legal information only. Consult a licensed attorney. ⚖️</div>")

        submit_btn.click(
            fn=process_legal_query,
            inputs=[question_input, context_input, disclaimer_checkbox, session_state],
            outputs=[response_output, debug_output, error_output, session_state]
        )
        clear_btn.click(fn=clear_form, inputs=[], outputs=[question_input, context_input, disclaimer_checkbox, response_output, debug_output])
        question_input.change(fn=lambda: "", inputs=[], outputs=[error_output])

    return interface

def main():
    print("🚀 Starting LawyerBot...")
    print("="*50)
    if not initialize_lawyerbot():
        print("❌ Failed to initialize LawyerBot. Check your .env file.")
        return

    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
