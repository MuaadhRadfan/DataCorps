from typing import Dict, Any, List, Union
from langchain_core.messages import BaseMessage, AIMessage
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import os
from dotenv import load_dotenv

load_dotenv()

class ALLaM:
    DEFAULT_MODEL_ID = "sdaia/allam-1-13b-instruct"
    DEFAULT_PARAMETERS = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 700,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.TEMPERATURE: 0.5,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 0.9,
        GenParams.REPETITION_PENALTY: 1,
        GenParams.STOP_SEQUENCES: ["\n\nHuman:", "\n\nAssistant:"]
    }

    def __init__(self, parameters: Dict[str, Any] = None):
        self.project_id = os.getenv("WATSONX_PROJECT_ID")
        self.api_key = os.getenv("WATSONX_APIKEY")
        
        # Initialize the model
        self.model = Model(
            model_id=self.DEFAULT_MODEL_ID,
            credentials={
                "apikey": self.api_key,
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id=self.project_id,
            params=self.DEFAULT_PARAMETERS
        )

    def invoke(self, messages: List[Union[Dict[str, str], BaseMessage]]) -> AIMessage:
        try:
            prompt = ""
            # Add a clear system message
            prompt += "You are a helpful AI assistant. Provide complete, detailed responses. Never cut off your response mid-sentence.\n\n"
            
            for message in messages:
                if isinstance(message, dict):
                    role = message["role"]
                    content = message["content"]
                else:
                    role = message.type
                    content = message.content
                
                if role == "system":
                    prompt += f"System: {content}\n"
                elif role == "user":
                    prompt += f"Human: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            
            prompt += "Assistant: "
            
            # Generate response
            response = self.model.generate(prompt=prompt)
            
            if response and response.get('results'):
                generated_text = response['results'][0]['generated_text']
                
                # Verify response isn't empty or truncated
                if generated_text and len(generated_text.strip()) > 0:
                    # Check if response ends properly
                    if not generated_text.strip().endswith(('.', '!', '?', '"', '؟', '.')):
                        generated_text = generated_text.strip() + '.'
                    return AIMessage(content=generated_text)
            
            raise Exception("Failed to get a valid response")
            
        except Exception as e:
            print(f"Error in invoke method: {str(e)}")
            raise e

allam = ALLaM()