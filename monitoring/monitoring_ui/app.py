import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/recent_delay_performance.png", overwrite=True)

with gr.Blocks() as demo:
  gr.Label("Recent performance history")
  input_img = gr.Image("recent_delay_performance.png", elem_id="predicted-img")

demo.launch()