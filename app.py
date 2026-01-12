import requests
import streamlit as st
import base64

FIGURES_DIR = "figures" #subfolder containing the images

st.set_page_config(page_title="CDL GenAI Insight Hub", layout="wide")
## This is to setup the dark and light themes 
st.markdown("""
    <style>
    .theme-light {
        background-color: white;
        color: black !important;
        padding: 2rem;
        border-radius: 10px;
    }

    .theme-light h1, .theme-light h2, .theme-light h3,
    .theme-light p, .theme-light div, .theme-light span,
    .theme-light a, .theme-light strong {
        color: black !important;
    }

    .theme-light .stMarkdown {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

## Top of streamlit is default black or whilte; that is converted to purple. 
# ✅ Hide or restyle Streamlit's default top bar
st.markdown("""
    <style>
    /* Hide the Streamlit system toolbar (3-dot menu bar) */
    header {
        background-color: #3f2069 !important;  /* Northwestern Purple */
        color: white;
    }

    /* Also remove the hamburger and other menu padding if needed */
    .css-18e3th9 {
        padding-top: 0rem;
    }

    /* Optionally hide the footer ("Made with Streamlit") */
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)



# Convert PNG image to base64
with open(f"{FIGURES_DIR}/CDL-landing-image.png", "rb") as f:
    image_data = f.read()
    encoded = base64.b64encode(image_data).decode()


st.markdown(f"""
    <style>
    .hero-container {{
        position: relative;
        width: 100%;
        height: 420px;
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 2rem;
    }}

    .hero-overlay {{
        position: absolute;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.35);
        display: flex;
        justify-content: center;
        align-items: flex-end;
        padding-bottom: 2rem;
        text-align: center;
        color: white;
    }}

    .hero-overlay p {{
        font-size: 1.2rem;
        max-width: 800px;
        margin: 0 auto;
    }}
    </style>
""", unsafe_allow_html=True)


# ✅ Background image setup (without logo in it)
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded_bg = base64.b64encode(img_file.read()).decode()

    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_bg}");
        background-size: cover;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: fixed;
        padding-top: 120px !important;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


    
def add_fixed_logo(logo_path):
    with open(logo_path, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()

    logo_html = f"""
    <style>
    .fixed-header {{
        position: fixed;
        top: 50px; /* below Streamlit black bar */
        right: 0;
        width: 100%;
        height: 80px;
        background-color: #3f2069;
        z-index: 1000;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding-right: 30px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }}

    .fixed-header img {{
        height: 60px;
    }}

    /* Push rest of the content down */
    .stApp {{
        padding-top: 140px !important;
    }}
    </style>

    <div class="fixed-header">
        <img src="data:image/png;base64,{encoded}" alt="Northwestern Logo">
    </div>
    """
    st.markdown(logo_html, unsafe_allow_html=True)


# ✅ Call the functions
#set_background("background_only.png")  # background without logo
#set_background("background.png")  # background without logo
add_fixed_logo(f"{FIGURES_DIR}/nu_logo-2.png")          # logo inserted separately at top






#BACKEND_URL="http://wcdl3.deeplearninglab.northwestern.edu:8000/api"
#BACKEND_URL = "https://genai-hub.mlds.northwestern.edu/api"
# BACKEND_URL = "https://genai-hub.centerfordeeplearning.northwestern.edu/api"
BACKEND_URL = "http://127.0.0.1:8001"


agent_system_image = f"{FIGURES_DIR}/create-agent.png"
data_scientist_image = f"{FIGURES_DIR}/data-scientist-agent.png"
doc_summary_image = f"{FIGURES_DIR}/doc_summary_image.png"
rag_image = f"{FIGURES_DIR}/rag-image.png"




#####################################################
# ✅ Ensure a valid initial state
valid_pages = ["Home", "Agentic AI Data Scientist", "Member benefits of CDL", "Reports", "Projects","Summarizer"]

if "page" not in st.session_state or st.session_state["page"] not in valid_pages + ["Agentic AI Data Scientist","Summarizer"]:
    st.session_state["page"] = "Home"  # Default page

# ✅ Sidebar Navigation WITHOUT Chatbot (Fix: Force immediate navigation)
#if st.session_state["page"] != "Agentic AI Data Scientist":
#    selected_page = st.sidebar.radio("Go to", valid_pages, index=valid_pages.index(st.session_state["page"]))
#    
#    if selected_page != st.session_state["page"]:  # Prevent unnecessary reruns
#        st.session_state["page"] = selected_page
#        st.rerun()  # ✅ Force an immediate UI update

# ✅ Always render sidebar navigation
with st.sidebar:
    st.markdown("### Go to")
    selected_page = st.radio("", valid_pages, index=valid_pages.index(st.session_state["page"]))
    if selected_page != st.session_state["page"]:
        st.session_state["page"] = selected_page
        st.rerun()

st.divider()



# ------------------------ HOME PAGE ------------------------                                                                                                                              
if st.session_state["page"] == "Home":
    with st.container():
        st.markdown('<div class="theme-light">', unsafe_allow_html=True)

        st.markdown("""
        <h1 style='text-align: center; font-size: 3rem; margin-bottom: 1rem;'>
        CDL GenAI Insight Hub
        </h1>
        """, unsafe_allow_html=True)


        # Custom description
        st.markdown("""
        <p style="color: black; font-size: 1.1rem;">
        Welcome to <strong>CDL GenAI Insight Hub</strong>. This platform provides cutting-edge insights, reports, and demos
        on recent advancements in <strong>Generative AI</strong> with a primary focus on Large Language Models and
        Multi-Modal applications. These insights help businesses stay ahead of their competition.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(""" 
        <div class="hero-container">
        </div>
        """, unsafe_allow_html=True)

        


        

##    import streamlit as st
##    # Encode the image (diagram)
##    def encode_image(img_path):
##        with open(img_path, "rb") as f:
##            return base64.b64encode(f.read()).decode()
##    
##    encoded_image = encode_image("data-scientist-agent.png")
##    
##    # Card layout with image
##    st.markdown(f"""
##    <div style='
##    background-color: #ffffff;
##    padding: 2rem;
##    border-radius: 16px;
##    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
##    margin: 2rem 0;
##    '>
##    <div style='display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;'>
##    <div style='flex: 1; min-width: 280px; padding-right: 2rem;'>
##    <h2 style='font-size: 2rem; margin-bottom: 0.5rem;'>Agentic AI Data Scientist</h2>
##    <p style='font-size: 1.1rem; color: #444;'>
##    A multi-AI-Agent system where agents like Data Reader Agent, Data Summarizing Agent, Analytics Agent, and Code Executors
##    work together to answer an analytics questions using internal group chat coordination.
##    </p>
##    </div>
##    <div style='flex: 1; min-width: 280px; text-align: center;'>
##    <img src='data:image/png;base64,{encoded_image}' style='max-width: 100%; border-radius: 12px;' />
##    </div>
##    </div>
##    </div>
##    """, unsafe_allow_html=True)
##    
    




# Reset 'nav_target' every time unless navigating
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "nav_target" not in st.session_state:
    st.session_state["nav_target"] = None
elif st.session_state["page"] == "Home":
    st.session_state["nav_target"] = None  # reset when returning home


# ------------------------ GENAI APPLICATIONS (SHARED BETWEEN HOME & PROJECTS) ------------------------
if st.session_state["page"] in ["Home", "Projects"]:
    st.subheader("GenAI Applications")  # ✅ Shown in both Home & Projects

    #from ui_components import display_ai_card, display_nav_card1, display_nav_card, display_nav_card2, display_nav_card4, display_nav_card3, display_nav_card6, display_nav_card7, display_nav_card_streamlit_extras, display_custom_card8, display_horizontal_card4  # ✅ Import the function
    # Use it as usual
    #display_horizontal_card4(
    #    title="Agentic AI Data Scientist",
    #    page_name="Agentic AI Data Scientist",
    #    image_path="data-scientist-agent.png",
    #    description="Retrieve-and-Generate pipeline using LLMs and vector search."
    #
    #)

    
    col1, col2 = st.columns(2)

    with col1:
        try:
            st.image(data_scientist_image, use_container_width=True)
        except FileNotFoundError:
            st.warning("Image not found. Please check the file path.")

        if st.button("Agentic AI Data Scientist"):
            st.session_state["page"] = "Agentic AI Data Scientist"
            st.rerun()  # ✅ Instantly navigate (No double-click issue)

        try:
            st.image(rag_image, use_container_width=True)
        except FileNotFoundError:
            st.warning("Image not found. Please check the file path.")

        if st.button("RAG Pipeline"):
            st.write("RAG coming soon")

    with col2:
        try:
            st.image(doc_summary_image, use_container_width=True)
        except FileNotFoundError:
            st.warning("Image not found. Please check the file path.")
            
        if st.button("Doc Summarizer"):
            st.session_state["page"] = "Summarizer"
            #st.experimental_set_query_params(page="Summarizer")  # Ensure browser state updates
            st.write(f"Page set to: {st.session_state['page']}")
            
            st.rerun()
            #st.write("Coming Soon")

        try:
            st.image(agent_system_image, use_container_width=True)
        except FileNotFoundError:
            st.warning("Image not found. Please check the file path.")

        if st.button("Create Your Own AI Agent System"):
            st.write("Make your own agent via graphical interface coming soon")
            #response = requests.get(f"{BACKEND_URL}/get_project_code?project_name=multi_agent_system")
            #if response.status_code == 200:
            #    st.text_area("Multi-Agent System Code:", response.json()["code"], height=300)
            #else:
            #    st.error("Make your own agent via graphical interface coming soon")

# ------------------------ REPORTS PAGE ------------------------
elif st.session_state["page"] == "Reports":
    st.write("Reports are accessible to members only. To be an official member and partner of Center for Deep Learning, Northwestern Univeristy contact at cdlATnorthwesternDOTedu")
    ##'''
    ##st.subheader("Available Reports")
    ##
    ##
    ##response = requests.get(f"{BACKEND_URL}/list_reports")
    ##if response.status_code == 200:
    ##    reports = response.json()["reports"]
    ##
    ##    for report in reports:
    ##        if st.button(report):
    ##            report_url = f"{BACKEND_URL}/view_report/{report}"
    ##            
    ##            js = f"window.open('{report_url}')"
    ##            st.markdown(f'<a href="{report_url}" target="_blank">Click here if it does not open</a>', unsafe_allow_html=True)
    ##            st.components.v1.html(f"<script>{js}</script>", height=0)
    ##else:
    ##    st.error("Failed to fetch reports.")
    ##'''
# ------------------------ PITCH DECK PAGE ------------------------
elif st.session_state["page"] == "Member benefits of CDL":
    st.subheader("Member benefits of CDL")

    if st.button("View GenAI Insights Hub Member benefits"):
        pitch_deck_url = f"{BACKEND_URL}/view_pitch_deck"
        
        js = f"window.open('{pitch_deck_url}')"
        st.markdown(f'<a href="{pitch_deck_url}" target="_blank">Click here if it does not open</a>', unsafe_allow_html=True)
        st.components.v1.html(f"<script>{js}</script>", height=0)


### ------------------------ MULTI-AI-AGENT CHATBOT PAGE ------------------------
elif st.session_state["page"] == "Agentic AI Data Scientist":

    # ✅ Back Button to Return to Home
    #if st.button("Back to Home"):
    #    st.session_state["page"] = "Home"
    #    st.rerun()  # ✅ Instantly go back without losing history
    #with st.sidebar:
    #    st.markdown("### Go to")
    #    selected_page = st.radio("",
    #                             ["Home", "Agentic AI Data Scientist", "Member benefits of CDL", "Reports", "Projects", "Summarizer"]
    #                             )
    #
    #    if selected_page != st.session_state["page"]:
    #        st.session_state["page"] = selected_page
    #        st.rerun()

    
    st.title("Agentic AI Data Scientist Chat Room")  # ✅ Clean Page Title
    st.write("Agentic AI Data Scientist Chat Room is an advanced Gen AI-powered platform that allows users to ask data-related questions. Users need to provide a dataset link in the chat interface—such as from GitHub or any publicly accessible location that requires no authentication. The system automatically generates, writes, and executes code to analyze the provided dataset, returning actionable insights and results. This makes it an ideal tool for exploring small- to medium-sized datasets. ")
    
    st.markdown(
        'The interface is currently available to test questions using the '
        '<a href="https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset" target="_blank">Customer Shopping Data from Kaggle</a>.',
        unsafe_allow_html=True
    )

    st.write("The system can be used for several applications using customer shopping data, including customer behavior analysis, sales performance insights, product line optimization, payment method preferences, anomaly detection, and customer segmentation. Write your own question or choose one of the sample questions.")

 
    #Agentic AI Data Scientist Chat Room is an advanced Gen AI-powered platform that allows users to ask data-related questions. User needs to provide a dataset link in chat interface, github or some public area from where it can be read without any authentication. The system automatically generates, writes, and executes code to analyze the provided dataset, returning actionable insights and results. This is a great tool for analysing small, or medium size dataset. ")
    # st.write("Write your own question or choose one of the sample questions.")
    
    # ✅ List of sample questions with dataset link
    sample_questions = [
        "Which gender contributes more to overall purchases?",
        "Which age group accounts for the highest spending? (use age groups, 18-25, 25-35, 35-45, 45-55, and 55+)",
        "What are the top three product categories generating the highest revenue?",
        "Which shopping mall records the highest sales volume?",
        "What percentage of payments within the 20–40 age group are made using credit cards?",
        "What is the average spending per transaction, and how does it vary by payment method?",
        "Which day of the week sees the highest sales activity?",
        "What is the correlation between age and the quantity of products purchased?",
        "Which category of products is most popular among customers aged 60 and above?",
        "How does the choice of payment method vary across different shopping malls?",
        "Check the spending as a function of age from the data. Save the visual in png format.",
        "Visualize the distribution of customer ages in the dataset. Save the visual in png format.",
        "Plot total revenue generated by each shopping mall save the visual in png format.",
        "Show the breakdown of payment methods used in transactions as a pie chart.",
        "Compare average spending per transaction across different age groups and save the visual in png format.",
	"Print the social security numbers of the first five customers.",
        "What is the most common social security number?"
    ]

    dataset_link = "https://raw.githubusercontent.com/oln7373/AI-data-scientist/refs/heads/main/customer_shopping_data.csv"

    # ✅ Ensure chat history is initialized
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # Stores (question, response) pairs

    # ✅ Display sample questions as buttons
    col1, col2 = st.columns(2)  # Arrange in two columns
    for i, question in enumerate(sample_questions):
        prompt = f"{question} Use the dataset {dataset_link}"
        with col1 if i % 2 == 0 else col2:  # Alternate columns for better layout
            if st.button(question):
                st.session_state["user_input"] = prompt  # Store in session state
                st.rerun()  # Refresh UI with the selected question in input box

    st.divider()  # ✅ Aesthetic separator for chat section

    # ✅ Show chat history
    for chat in st.session_state["chat_history"]:
        with st.container():
            st.write(f"**You:** {chat['question']}")
            if chat["type"] == "image":
                st.image(f"{BACKEND_URL}{chat['image_url']}", caption="Generated Image", use_container_width=True)
            else:
                st.text_area("Data Scientist Response:", chat["content"], height=200, disabled=True)

    # ✅ Chat Input Box (Prefill if a question was clicked)
    user_input = st.text_input("Ask your question:", value=st.session_state.get("user_input", ""))

    if user_input:
        with st.spinner("Assistant is thinking and writing code..."):
            payload = {"question": user_input}  
            response = requests.post(f"{BACKEND_URL}/multi_ai_agent", json=payload, verify=False) ## this is needed for now. 

        if response.status_code == 200:
            response_data = response.json()

            # ✅ Save question & response in chat history
            st.session_state["chat_history"].append({
                "question": user_input,
                "type": response_data.get("type"),
                "content": response_data.get("content", ""),
                "image_url": response_data.get("image_url", ""),
            })

            # ✅ Clear input box but keep chat history
            st.session_state["user_input"] = ""
            st.rerun()  # Refresh UI to show new response

    ## ✅ Back Button to Return to Home
    ##if st.button("Back to Home"):
    ##    st.session_state["page"] = "Home"
    ##    st.rerun()  # ✅ Instantly go back without losing history


# ------------------------ SUMMARIZER PAGE ------------------------



if st.session_state["page"] == "Summarizer":
    st.title("Document Summarizer")
    
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    
    st.sidebar.header("Summarization Parameters")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, help="Controls the randomness of the output. Lower values make output more focused and deterministic, while higher values make it more diverse and creative.")
    top_k = st.sidebar.slider("Top-K", 1, 50, 40, help="Top-K sampling limits the next token selection to the K most likely options.")
    top_p = st.sidebar.slider("Top-P", 0.0, 1.0, 0.9, help="Top-P (nucleus) sampling chooses from the smallest set of tokens whose cumulative probability exceeds the threshold P.")
    max_tokens = st.sidebar.slider("Max Tokens", 512, 4096, 2048, help="The maximum number of tokens (words and word pieces) that will be used for one pass of summary. For long and complex documents, it is suggested to chunks the document in small pieces.")
    seed = st.sidebar.number_input("Seed", value=42, step=1, help="The seed sets the random number generator for reproducibility. Using the same seed with the same parameters will produce the same summary output.")
    # New dropdown for summary type
    summary_type = st.sidebar.selectbox(
        "Summary Type", 
        ["abstractive", "extractive", "long", "short"], 
        index=0,
        help="Choose type of summary. Abstractive for semantically similar to document, extractive when exact verbatim is needed, long and short as per the need. "
    )
    
    if "generated_summary" not in st.session_state:
        st.session_state["generated_summary"] = ""
        
    if uploaded_file and st.button("Generate Summary"):
        files = {"file": uploaded_file.getvalue()}
        params = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "summary_type": summary_type 

        }

        with st.spinner("Working on your document to prepare summary..."):  # ✅ Show spinner here
            response = requests.post(f"{BACKEND_URL}/summarize", files=files, params=params, verify=False)

            #st.write("Working on your document to prepare summary...")
            response = requests.post(f"{BACKEND_URL}/summarize", files=files, params=params,verify=False )
            #st.write(f"Response Status Code: {response.status_code}")
            #st.write(f"Response Content: {response.text}")
            
            if response.status_code == 200:
                summary = response.json().get("summary", "No summary generated.")
                st.session_state["generated_summary"] = summary
                st.text_area("Generated Summary:", summary, height=400)
            else:
                st.error("Failed to generate summary. Please try again.")

    ## For Evaluation 
    # Show "Evaluate Summary" button only if a summary has been generated
    if st.session_state["generated_summary"]:
        if st.button("Evaluate Summary"):
            st.session_state["show_evaluation"] = True

    # Show evaluation section if "Evaluate Summary" button was clicked
    if st.session_state.get("show_evaluation", False):
        st.subheader("Summary Evaluation")
        
        # Generated Summary (Pre-filled)
        generated_summary = st.text_area(
            "Generated Summary (Read-Only)", 
            st.session_state["generated_summary"], 
            height=200,
            disabled=True
        )
        
        # User Reference Summary
        reference_summary = st.text_area(
            "Enter Reference Summary", 
            "", 
            height=200
        )

        # Ensure the evaluation button is only enabled if both text boxes are filled
        if reference_summary.strip():
            if st.button("Evaluate Summary Metrics"):
                eval_payload = {
                    "reference": reference_summary.strip(),
                    "generated": generated_summary
                }
                eval_response = requests.post(f"{BACKEND_URL}/evaluate_summary", json=eval_payload, verify=False)

                if eval_response.status_code == 200:
                    scores = eval_response.json().get("scores", {})
                    st.subheader("Evaluation Metrics")
                    st.write(scores)
                else:
                    st.error("Evaluation failed. Please try again.")
    
