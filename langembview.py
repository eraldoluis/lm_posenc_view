from transformers import AutoModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

#
# Figure Styling
#

# color map (dark around zero, bright on both extremes)
cmap='twilight'

# figure size and resolution
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams["figure.dpi"] = 300

#
# Load Models
#

@st.cache
def load_wpe_gpt():
    '''
    Load word position embeddings from GPT-2 model.

    This is a cached Streamlit function.
    '''
    gpt2 = AutoModel.from_pretrained('gpt2-xl')
    return gpt2.wpe.weight.detach().numpy()


@st.cache
def load_wpe_bert():
    '''
    Load word position embeddings from BERT model.

    This is a cached Streamlit function.
    '''
    bert = AutoModel.from_pretrained('bert-large-uncased')
    return bert.embeddings.position_embeddings.weight.detach().numpy()


wpe_gpt = load_wpe_gpt()
wpe_bert = load_wpe_bert()

'# Shapes'

'## GPT-2'
wpe_gpt.shape

'## BERT'
wpe_bert.shape


'# Statistics'

# '## GPT-2'
# st.write(pd.DataFrame(wpe_gpt.flatten()).describe())

'## BERT'
st.write(pd.DataFrame(wpe_bert.flatten()).describe())


'# Histograms'

# '## GPT-2'
# fig, ax = plt.subplots()
# ax.hist(wpe_gpt.flatten(), bins=1000, density=True)
# st.pyplot(fig)

# fig, ax = plt.subplots()
# ax.hist(wpe_gpt.flatten(), bins=1000, range=[-0.05, 0.05], density=True)
# st.pyplot(fig)

'## BERT'
fig, ax = plt.subplots()
ax.hist(wpe_bert.flatten(), bins=1000, density=True)
st.pyplot(fig)

fig, ax = plt.subplots()
ax.hist(wpe_bert.flatten(), bins=1000, range=[-0.1, 0.1], density=True)
st.pyplot(fig)


'# Visualizations of Positional Encodings'

# '## GPT-2'
# fig, ax = plt.subplots()
# ax.imshow(wpe_gpt, vmin=-0.05, vmax=0.05, interpolation="none", origin='lower', cmap=cmap)
# ax.yticks([0,256,512,768,1024])
# ax.xlabel('Encoding dimension')
# ax.ylabel('Position')
# ax.title('GPT-2 Positional Encodings')
# st.pyplot(fig)

'## BERT'
fig, ax = plt.subplots()
ax.imshow(wpe_bert, vmin=-.12, vmax=.12, interpolation="none", origin='lower', cmap=cmap)
ax.set_yticks([0,128,256,384,512])
ax.set_xlabel('Encoding dimension')
ax.set_ylabel('Position')
ax.set_title('BERT Positional Encondings')
st.pyplot(fig)

'## Sorted by 2-Norm'

# '### GPT-2'
# # sort cols by 2-norm (descending)
# wpe_gpt_sort_idxs = np.argsort(-np.linalg.norm(wpe_gpt, axis=0))
# wpe_gpt_sort = wpe_gpt[:,wpe_gpt_sort_idxs]
# # plot
# fig, ax = plt.subplots()
# ax.imshow(wpe_gpt_sort, vmin=-0.05, vmax=0.05, interpolation="none", origin='lower', cmap=cmap)
# ax.yticks([0,256,512,758,1024])
# ax.xlabel('Encoding dimension')
# ax.ylabel('Position')
# ax.title('GPT-2 Positional Encodings (sorted)')
# st.pyplot(fig)

'### BERT'
# sort cols by maximum abolute value (descending)
wpe_bert_sort_idxs = np.argsort(-np.linalg.norm(wpe_bert, axis=0))
wpe_bert_sort = wpe_bert[:,wpe_bert_sort_idxs]
# plot
fig, ax = plt.subplots()
ax.imshow(wpe_bert_sort, vmin=-0.12, vmax=0.12, interpolation="none", origin='lower', cmap=cmap)
ax.set_yticks([0,128,256,384,512])
ax.set_xlabel('Encoding dimension')
ax.set_ylabel('Position')
ax.set_title('BERT Positional Encondings (sorted)')
st.pyplot(fig)
