import streamlit as st
from typing import List
from chatcomponent import ChatComponent
from componentresultobject import ComponentResultObject
import ollama


class ExkimoStreamlitApp:
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Exkimo Bot",
            page_icon="🤖",
            layout="centered"
        )
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        if "available_models" not in st.session_state:
            st.session_state.available_models = self.get_available_models()
            
        if "selected_model" not in st.session_state:
            if st.session_state.available_models:
                st.session_state.selected_model = st.session_state.available_models[0]
            else:
                st.session_state.selected_model = "gemma3:4b"
            
        if "chat_component" not in st.session_state:
            st.session_state.chat_component = ChatComponent(
                language_model=f"ollama:{st.session_state.selected_model}",
                temperature=0.0
            )
            
        if "file_content" not in st.session_state:
            st.session_state.file_content = None
            
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            models = ollama.list()
            return [model["model"] for model in models["models"]]
        except Exception:
            return []
            
    def update_model(self, model_name: str):
        """Update the chat component with a new model"""
        st.session_state.selected_model = model_name
        st.session_state.chat_component = ChatComponent(
            language_model=f"ollama:{model_name}",
            temperature=0.0
        )
        
    def reset_chat(self):
        """Reset chat history and file content"""
        st.session_state.messages = []
        st.session_state.file_content = None
        
    def create_message_object(self, role: str, content: str) -> ComponentResultObject:
        """Create a ComponentResultObject for a message"""
        msg = ComponentResultObject()
        msg["source"] = role
        msg["content"]["original_text"] = content
        return msg
        
    def prepare_system_prompt(self) -> ComponentResultObject:
        """Prepare system prompt with file content if available"""
        system_msg = ComponentResultObject()
        system_msg["source"] = "system"
        
        if st.session_state.file_content:
            system_msg["content"]["original_text"] = f"Nutze folgende Information als Kontext: {st.session_state.file_content}"
        else:
            system_msg["content"]["original_text"] = "Du bist ein hilfreicher Assistent."
            
        return system_msg
        
    def display_chat_history(self):
        """Display all messages in the chat history"""
        for message in st.session_state.messages:
            with st.chat_message(message["source"]):
                st.write(message["content"]["original_text"])
                
    def handle_file_upload(self, uploaded_file):
        """Process uploaded text file"""
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode("utf-8")
                st.session_state.file_content = content
                st.success(f"Datei '{uploaded_file.name}' erfolgreich geladen!")
            except Exception as e:
                st.error(f"Fehler beim Laden der Datei: {str(e)}")
                
    def run(self):
        """Main application logic"""
        # Header
        st.title("🤖 Exkimo Bot")
        st.markdown("---")
        
        # Model selection section
        with st.container():
            st.subheader("🎯 Modell-Auswahl")
            
            if st.session_state.available_models:
                selected_model = st.selectbox(
                    "Wähle ein Ollama-Modell:",
                    options=st.session_state.available_models,
                    index=st.session_state.available_models.index(st.session_state.selected_model)
                    if st.session_state.selected_model in st.session_state.available_models
                    else 0,
                    key="model_selector"
                )
                
                # Update model if selection changed
                if selected_model != st.session_state.selected_model:
                    self.update_model(selected_model)
                    st.success(f"Modell gewechselt zu: {selected_model}")
            else:
                st.warning("Keine Ollama-Modelle gefunden. Stelle sicher, dass Ollama läuft und Modelle installiert sind.")
                st.info("Installiere Modelle mit: `ollama pull model_name`")
        
        # File upload section
        with st.container():
            st.subheader("📄 Datei-Upload (Optional)")
            uploaded_file = st.file_uploader(
                "Lade eine Textdatei hoch, die als Kontext verwendet werden soll",
                type=['txt'],
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                self.handle_file_upload(uploaded_file)
                
            if st.session_state.file_content:
                with st.expander("Geladener Dateiinhalt anzeigen"):
                    st.text(st.session_state.file_content[:500] + "..." 
                           if len(st.session_state.file_content) > 500 
                           else st.session_state.file_content)
                    
        # New chat button
        if st.button("🔄 Neuer Chat", use_container_width=True):
            self.reset_chat()
            st.rerun()
            
        # Chat container
        st.markdown("---")
        st.subheader("💬 Chat")
        
        # Display chat history
        self.display_chat_history()
        
        # Chat input
        if prompt := st.chat_input("Schreibe eine Nachricht..."):
            # Add user message to chat
            user_msg = self.create_message_object("user", prompt)
            st.session_state.messages.append(user_msg)
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
                
            # Generate and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Convert messages to ComponentResultObject format
                chat_history = []
                
                # Add system prompt as first message
                system_prompt = self.prepare_system_prompt()
                chat_history.append(system_prompt)
                
                # Add all chat messages
                for msg in st.session_state.messages:
                    chat_history.append(self.create_message_object(
                        msg["source"], 
                        msg["content"]["original_text"]
                    ))
                
                # Stream the response
                try:
                    for chunk in st.session_state.chat_component.stream(chat_history):
                        if chunk.get('message', {}).get('content'):
                            full_response += chunk['message']['content']
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant message to history
                    assistant_msg = self.create_message_object("assistant", full_response)
                    st.session_state.messages.append(assistant_msg)
                    
                except Exception as e:
                    st.error(f"Fehler bei der Antwortgenerierung: {str(e)}")
                    st.info("Stelle sicher, dass Ollama läuft und das Modell 'gemma3:4b' installiert ist.")
                    
        # Footer
        st.markdown("---")
        st.caption("Exkimo Bot - Powered by Ollama")
