#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // When the arm override is active the real hardware arms are at the VLA target
    // pose, but the policy was not trained with arms up.  Spoof the arm joint
    // observations with default positions so the policy keeps the legs stable.
    env->post_update_hook = [this]() {
        if (FSMState::arm_override_state_ == FSMState::ArmOverrideState::OFF) return;
        auto& data = env->robot->data;
        for (int i = 0; i < static_cast<int>(data.joint_ids_map.size()); i++) {
            int hw_idx = static_cast<int>(data.joint_ids_map[i]);
            for (int arm_hw : FSMState::ARM_JOINT_INDICES) {
                if (hw_idx == arm_hw) {
                    data.joint_pos[i] = data.default_joint_pos[i];
                    data.joint_vel[i] = 0.0f;
                    break;
                }
            }
        }
    };

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}