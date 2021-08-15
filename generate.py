import os
import warnings
import requests
import torch
import streamlit as st
from streamlit_lottie import st_lottie
from transformers import AutoTokenizer, AutoModelWithLMHead

warnings.filterwarnings("ignore")

st.set_page_config(layout='centered', page_title='GPT2-Horoscopes')

def load_lottieurl(url: str):
    # https://github.com/tylerjrichards/streamlit_goodreads_app/blob/master/books.py
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_book = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_WL3aE7.json')
st_lottie(lottie_book, speed=1, height=200, key="initial")

st.markdown('# GPT2-Horoscopes!')
st.markdown("""
Hello! This lovely app lets GPT-2 write awesome horoscopes for you. All you need to do
is select your sign and choose the horoscope category :)  
""")
st.markdown("""
*If you are interested in the fine-tuned model, you can visit the [Model Hub](https://huggingface.co/shahp7575/gpt2-horoscopes) or 
my [GitHub Repo](https://github.com/shahp7575/gpt2-horoscopes).*
""")


@st.cache(allow_output_mutation=True, max_entries=1)
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

    horoscope = st.selectbox("Choose Your Sign: ", ('Aquarius', 'Pisces', 'Aries',
                                                         'Taurus', 'Gemini', 'Cancer',
                                                         'Leo', 'Virgo', 'Libra', 
                                                          'Scorpio', 'Sagittarius', 'Capricorn'), index=0)
    choice = st.selectbox("Choose Category:", ('general', 'career', 'love', 'wellness', 'birthday'),
                                    index=0, )

    temp_slider = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.95)

if st.button('Generate Horoscopes!'):
    prompt = make_prompt(choice)
    prompt_encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    with st.spinner('Generating...'):
        sample_output = generate(prompt_encoded, model, tokenizer, temperature=0.95, num_outputs=1, top_k=40)
        final_out = tokenizer.decode(sample_output[0], skip_special_tokens=True)
        st.write(final_out[len(choice)+2:])
else: pass


