import pickle

embedding_file_20_path = "/home/aidrive/zhengxj/projects_new/MTR_new/LLM_integrate/LLM_output/context_file/llm_output_context_with_encoder_20.pkl"
embedding_file_100_path = "/home/aidrive/zhengxj/projects_new/MTR_new/LLM_integrate/LLM_output/context_file/llm_output_context_with_encoder_100.pkl"

context_data_train_path = "/home/aidrive/zhengxj/projects_new/MTR_new/LLM_integrate/context_data/train/context_data_encoder_100.pkl"
context_data_test_path = "/home/aidrive/zhengxj/projects_new/MTR_new/LLM_integrate/context_data/test/context_data_encoder_100.pkl"
context_data_valid_path = "/home/aidrive/zhengxj/projects_new/MTR_new/LLM_integrate/context_data/valid/context_data_encoder_100.pkl"

embedding_file_list = [embedding_file_20_path, embedding_file_100_path]
context_data_list = [context_data_train_path, context_data_valid_path, context_data_test_path]

agent_type_list = ["TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"]

# for embedding_file_path in embedding_file_list:
#     data = pickle.load(open(embedding_file_path, 'rb'))
#     for agent_type in agent_type_list:
#         data_for_certain_type = data[agent_type]
#         intention_list = data_for_certain_type["intention"]
#         affordance_list = data_for_certain_type["affordance"]
#         scenario_list = data_for_certain_type["scenario"]
#         agent_num = len(scenario_list)
#         for agent_index in range(agent_num):
#             intention_list[agent_index] = [intention for intention in intention_list[agent_index] if intention != ""]
#             affordance_list[agent_index] = [affordance for affordance in affordance_list[agent_index] if affordance != ""]
#             scenario_list[agent_index] = [scenario for scenario in scenario_list[agent_index] if scenario != ""]
#     pickle.dump(data, open(embedding_file_path+"new", 'wb'))
            
for context_file_path in context_data_list:
    data = pickle.load(open(context_file_path, 'rb'))
    for scenario_id, scenario_context in data.items():
        agent_num = len(scenario_context["track_indexes"])
        for agent_index in range(agent_num):
            agent_track_intentions = data[scenario_id]["track_intentions"][agent_index]
            agent_track_affordances = data[scenario_id]["track_affordances"][agent_index]
            agent_track_scenarios = data[scenario_id]["track_scenarios"][agent_index]
            for context_index in range(4):
                agent_track_intentions[context_index] = [intention for intention in agent_track_intentions[context_index] if intention != ""]
                agent_track_affordances[context_index] = [affordance for affordance in agent_track_affordances[context_index] if affordance != ""]
                agent_track_scenarios[context_index] = [scenario for scenario in agent_track_scenarios[context_index] if scenario != ""]
    pickle.dump(data, open(context_file_path + "new", 'wb'))