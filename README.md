Link to whole_quantized_model.pt:
https://iiitaphyd-my.sharepoint.com/:u:/g/personal/soham_ghosh_students_iiit_ac_in/Ebeby2cWnx5NvJbVj24oNP0BOyhCQyTmFdRwvsOZb4hRMw?e=UxOI9M

Link to selectively_quantized_model:
https://iiitaphyd-my.sharepoint.com/:u:/g/personal/soham_ghosh_students_iiit_ac_in/EUBHJVHMSq5MvNNMx53zVSABNczH1OutuuVygtyy5Zzauw?e=qIWDpq

Link to bnb_8bit_quantized_model.pt:
https://iiitaphyd-my.sharepoint.com/:u:/g/personal/soham_ghosh_students_iiit_ac_in/ER77cX-oDIpGnsqfFKcvbewBaYuTTWQdaznGhrYSSiqd3g?e=DPYrpp

Link to bnb_4bit_quantized_model.pt:
https://iiitaphyd-my.sharepoint.com/:u:/g/personal/soham_ghosh_students_iiit_ac_in/EaW-g-1ceVlDpuxK4XrHqp0BHxWTByWiMPI-RB6PMmI6hQ?e=6Lmlqb

Link to bnb_nf4_quantized_model.pt:
https://iiitaphyd-my.sharepoint.com/:u:/g/personal/soham_ghosh_students_iiit_ac_in/EfC4QAF73y9Kmv5ux0BRsLEB_YSFsTuf0thzaGISLQG1Tg?e=RtxlMp

For loading any of the above quantized models:
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.load_state_dict(torch.load(<quantized_model_path>))

The necessary libraries and packages must be installed before running the python files