import os
import config
import warnings
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelWithLMHead

warnings.filterwarnings("ignore")

st.header("Get Your Horoscopes!!")
st.write("""
Created by Parth Shah

This is a Streamlit web app!
""")

@st.cache(allow_output_mutation=True)
def download_model():
    tokenizer = AutoTokenizer.from_pretrained('shahp7575/gpt2-horoscopes')
    model = AutoModelWithLMHead.from_pretrained('shahp7575/gpt2-horoscopes')
    return model, tokenizer
model, tokenizer = download_model()

def make_prompt(category):
    return f"<|category|> {category} <|horoscope|>"

def generate(prompt, model, tokenizer, temperature, num_outputs, top_k):

    sample_outputs = model.generate(prompt, 
                                    #bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=top_k, 
                                    max_length = 300,
                                    top_p=0.95,
                                    temperature=temperature,
                                    num_return_sequences=num_outputs)

    return sample_outputs
    
with st.beta_container():

    choice = st.selectbox("Choose Category:", ('general', 'career', 'love', 'wellness', 'birthday'),
                                    index=0, )
    horoscope = st.selectbox("Choose Your Horoscope: ", ('aries', 'etc'))

if st.button('Generate Horoscopes!'):        
    with st.spinner('Generating...'):
        prompt = make_prompt(choice)
        prompt_encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

        sample_output = generate(prompt_encoded, model, tokenizer, temperature=0.95, num_outputs=1, top_k=40)
        final_out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
        st.write(final_out[len(choice)+2:])
else: pass


