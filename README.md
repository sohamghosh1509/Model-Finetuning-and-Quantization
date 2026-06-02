Link to soft_prompt_model.pt: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/soham_ghosh_students_iiit_ac_in/EWh7gxoFWTpHmHJQkkXE9TABMgrTZxBp9YFttGXxxlpH1Q?e=NhRL5e

For restoring the above trained model, run the following code:
model = GPT2LMHeadModel.from_pretrained("gpt2")
soft_prompt_model = SoftPromptTuning(model)
soft_prompt_model.load_state_dict(torch.load("soft_prompt_model.pt"))

Link to lora_finetuned_model.pt: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/soham_ghosh_students_iiit_ac_in/ERTWW9y5n1xPmRRO4JZs6QQBHW4fDHcvjHw2G73tfcs4xg?e=vv9bdu

For restoring the above trained model, run the following code:
model = GPT2LMHeadModel.from_pretrained("gpt2")
lora_model = get_peft_model(model, lora_config)
lora_model.load_state_dict(torch.load("lora_finetuned_model.pt"))

Link to fintuned_model.pt: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/soham_ghosh_students_iiit_ac_in/EXoaONKz_R5CpO-suSWvno8Bp75edRFa7n6mZN1HPSe7GA?e=IRpMJZ

For restoring the above trained model, run the following code:
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.load_state_dict(torch.load("finetuned_model.pt"))

The necessary libraries and packages must be installed before running the python files

The number of epochs has been taken a low value due to GPU quota constraints on Kaggle