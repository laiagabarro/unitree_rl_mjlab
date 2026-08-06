#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "keyboard_input.h"
#include "unitree_joystick_dsl_extended.hpp"

// Controller arbitration: allows seamless switching with g1_deploy_onnx_ref.
// This controller is ID=1 and is inactive by default.
#include <atomic>
#include <memory>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/ros2/String_.hpp>

static constexpr const char* CONTROLLER_SWITCH_TOPIC_FSM       = "rt/controller_switch";
static constexpr const char* CONTROLLER_SWITCH_READY_TOPIC_FSM = "rt/controller_switch_ready";

/**
 * @brief Controller arbitration gate for g1_ctrl (controller ID=1).
 *
 * Activation is triggered by the *ready* topic (rt/controller_switch_ready),
 * not the intent topic (rt/controller_switch).  This ensures g1_ctrl only
 * starts publishing after g1_deploy_onnx_ref has finished its deactivation
 * animation and explicitly signalled the handoff is complete.
 */
class ControllerSwitchGate {
public:
    explicit ControllerSwitchGate(int my_id, bool default_active)
        : my_id_(my_id), active_(default_active)
    {
        // Subscribe to the ready topic for the activation edge.
        ready_sub_.reset(new unitree::robot::ChannelSubscriber<std_msgs::msg::dds_::String_>(
            CONTROLLER_SWITCH_READY_TOPIC_FSM));
        ready_sub_->InitChannel(
            [this](const void* msg) {
                const auto& str = static_cast<const std_msgs::msg::dds_::String_*>(msg)->data();
                int id = -1;
                try { id = std::stoi(str); } catch (...) { return; }
                bool was_active = active_.load(std::memory_order_relaxed);
                bool now_active = (id == my_id_);
                if (!was_active && now_active) {
                    active_.store(true, std::memory_order_release);
                    just_activated_.store(true, std::memory_order_release);
                }
            }, 1);

        // Subscribe to the intent topic only to detect deactivation requests
        // (operator sends "0" to hand control back to g1_deploy_onnx_ref).
        intent_sub_.reset(new unitree::robot::ChannelSubscriber<std_msgs::msg::dds_::String_>(
            CONTROLLER_SWITCH_TOPIC_FSM));
        intent_sub_->InitChannel(
            [this](const void* msg) {
                const auto& str = static_cast<const std_msgs::msg::dds_::String_*>(msg)->data();
                int id = -1;
                try { id = std::stoi(str); } catch (...) { return; }
                bool was_active = active_.load(std::memory_order_relaxed);
                bool now_active = (id == my_id_);
                // Only handle deactivation here; activation comes via the ready topic.
                // The caller must call complete_deactivation() to publish the ready signal.
                if (was_active && !now_active) {
                    deactivation_pending_.store(true, std::memory_order_release);
                }
            }, 1);
    }

    bool is_active() const noexcept { return active_.load(std::memory_order_acquire); }

    /// Returns true exactly once when the ready signal arrives for this controller.
    bool consume_activation() noexcept { return just_activated_.exchange(false, std::memory_order_acq_rel); }

    /// Returns true exactly once when a deactivation request arrives on the intent topic.
    /// is_active() remains true until complete_deactivation() is called.
    bool consume_deactivation_pending() noexcept { return deactivation_pending_.exchange(false, std::memory_order_acq_rel); }

    /// Call when ready to yield lowcmd.  Flips gate→false and publishes the ready
    /// signal (incoming controller id = 1 - my_id_) so the other side can activate.
    void complete_deactivation() {
        ensure_ready_publisher();
        std_msgs::msg::dds_::String_ msg;
        msg.data() = std::to_string(1 - my_id_);
        // Publish the ready signal BEFORE yielding the gate so the incoming
        // controller can start before we stop — eliminating the lowcmd gap.
        // Repeat 3× to guard against a single dropped DDS message.
        ready_pub_->Write(msg);
        ready_pub_->Write(msg);
        ready_pub_->Write(msg);
        active_.store(false, std::memory_order_release);
        just_deactivated_.store(true, std::memory_order_release);
    }

    /// Returns true exactly once after complete_deactivation().
    bool consume_deactivation() noexcept { return just_deactivated_.exchange(false, std::memory_order_acq_rel); }

private:
    void ensure_ready_publisher() {
        if (ready_pub_) return;
        ready_pub_.reset(
            new unitree::robot::ChannelPublisher<std_msgs::msg::dds_::String_>(
                CONTROLLER_SWITCH_READY_TOPIC_FSM));
        ready_pub_->InitChannel();
    }

    int my_id_;
    std::atomic<bool> active_;
    std::atomic<bool> just_activated_{false};
    std::atomic<bool> just_deactivated_{false};
    std::atomic<bool> deactivation_pending_{false};
    std::shared_ptr<unitree::robot::ChannelSubscriber<std_msgs::msg::dds_::String_>> ready_sub_;
    std::shared_ptr<unitree::robot::ChannelSubscriber<std_msgs::msg::dds_::String_>> intent_sub_;
    std::shared_ptr<unitree::robot::ChannelPublisher<std_msgs::msg::dds_::String_>>  ready_pub_;
};

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                auto func = unitree::common::dsl::CompileExpressionExtended(condition);
                registered_checks.emplace_back(std::make_pair(
                    [func]() -> bool
                    {
                        unitree::common::dsl::InputContext ctx;
                        ctx.joystick = &FSMState::lowstate->joystick;
                        ctx.keyboard = FSMState::keyboard_input.get();
                        return func(ctx);
                    },
                    fsm_id));
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void enter() override
    {
        // If the previous state requested no interpolation (e.g. a dance/Mimic policy
        // whose joint positions are already close to the neutral pose), skip the blend.
        if (skip_next_enter_interp_) {
            skip_next_enter_interp_ = false;
            activation_interp_active_ = false;
            activation_interp_frame_ = 0;
            spdlog::info("FSM enter {}: skipping upper-body blend (Mimic exit)",
                         getStateString());
            return;
        }
        // Capture current hardware positions and blend the upper body into the new policy.
        for (int i = G1_UPPER_BODY_START; i < G1_TOTAL_JOINTS; i++) {
            activation_interp_start_[i] = static_cast<double>(
                lowstate->msg_.motor_state()[i].q());
        }
        activation_interp_active_ = true;
        activation_interp_frame_ = 0;
        spdlog::info("FSM enter {}: blending upper body over {} frames",
                     getStateString(), ACTIVATION_INTERP_FRAMES);
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard)
            keyboard->update();
        if (keyboard_input)
            keyboard_input->update();
    }

    void post_run()
    {
        // Activation: g1_deploy_onnx_ref has finished its deactivation animation and
        // published the ready signal.  Capture current hardware positions and blend in.
        if (controller_switch_ && controller_switch_->consume_activation()) {
            for (int i = 0; i < G1_TOTAL_JOINTS; i++) {
                activation_interp_start_[i] = static_cast<double>(
                    lowstate->msg_.motor_state()[i].q());
            }
            activation_interp_active_ = true;
            activation_interp_frame_ = 0;
            lower_body_interp_active_ = false;
            lower_body_interp_frame_ = 0;
            spdlog::info("Controller activated: interpolating upper body over {} frames, lower body over {} frames",
                         ACTIVATION_INTERP_FRAMES, LOWER_BODY_INTERP_FRAMES);
        }

        if (lower_body_interp_active_) {
            double t     = std::min(1.0, static_cast<double>(lower_body_interp_frame_) /
                                         static_cast<double>(LOWER_BODY_INTERP_FRAMES));
            double alpha = t * t * (3.0 - 2.0 * t);
            for (int i = 0; i < G1_UPPER_BODY_START; i++) {
                double target = static_cast<double>(lowcmd->msg_.motor_cmd()[i].q());
                lowcmd->msg_.motor_cmd()[i].q() = static_cast<float>(
                    activation_interp_start_[i] * (1.0 - alpha) + target * alpha);
            }
            lower_body_interp_frame_++;
            if (lower_body_interp_frame_ >= LOWER_BODY_INTERP_FRAMES) {
                lower_body_interp_active_ = false;
                spdlog::info("Lower-body activation blend complete.");
            }
        }

        if (activation_interp_active_) {
            // Smoothstep (ease-in-out cubic) matches the blend used by g1_deploy_onnx_ref.
            double t     = std::min(1.0, static_cast<double>(activation_interp_frame_) /
                                         static_cast<double>(ACTIVATION_INTERP_FRAMES));
            double alpha = t * t * (3.0 - 2.0 * t);
            for (int i = G1_UPPER_BODY_START; i < G1_TOTAL_JOINTS; i++) {
                double target = static_cast<double>(lowcmd->msg_.motor_cmd()[i].q());
                lowcmd->msg_.motor_cmd()[i].q() = static_cast<float>(
                    activation_interp_start_[i] * (1.0 - alpha) + target * alpha);
            }
            activation_interp_frame_++;
            if (activation_interp_frame_ >= ACTIVATION_INTERP_FRAMES) {
                activation_interp_active_ = false;
                spdlog::info("Activation interpolation complete! Controller fully active.");
            }
        }

        // Deactivation: operator sent "0" — publish ready signal immediately so
        // g1_deploy_onnx_ref can activate and start its blend-in.
        if (controller_switch_ && controller_switch_->consume_deactivation_pending()) {
            activation_interp_active_ = false;
            lower_body_interp_active_ = false;
            arm_override_state_ = ArmOverrideState::OFF;
            controller_switch_->complete_deactivation();  // publishes ready "0"
            spdlog::info("Controller deactivated: published ready signal, yielding lowcmd to g1_deploy_onnx_ref.");
        }

        // Arm override: up-arrow → blend to VLA ready pose; down-arrow → blend back to policy.
        if (keyboard_input) {
            if (keyboard_input->key_up.on_pressed &&
                arm_override_state_ != ArmOverrideState::BLEND_TO_VLA &&
                arm_override_state_ != ArmOverrideState::HOLD_VLA) {
                // Capture current hardware positions as blend start.
                for (int k = 0; k < G1_23DOF_ARM_JOINT_COUNT; k++) {
                    arm_override_interp_start_[k] = static_cast<double>(
                        lowstate->msg_.motor_state()[ARM_JOINT_INDICES[k]].q());
                }
                arm_override_state_ = ArmOverrideState::BLEND_TO_VLA;
                arm_override_frame_ = 0;
                spdlog::info("Arm override: blending to VLA ready pose over {} frames", ARM_INTERP_FRAMES);
            }
            if (keyboard_input->key_down.on_pressed &&
                arm_override_state_ != ArmOverrideState::OFF &&
                arm_override_state_ != ArmOverrideState::BLEND_TO_POLICY) {
                // Capture current hardware positions as blend start.
                for (int k = 0; k < G1_23DOF_ARM_JOINT_COUNT; k++) {
                    arm_override_interp_start_[k] = static_cast<double>(
                        lowstate->msg_.motor_state()[ARM_JOINT_INDICES[k]].q());
                }
                arm_override_state_ = ArmOverrideState::BLEND_TO_POLICY;
                arm_override_frame_ = 0;
                spdlog::info("Arm override: blending back to policy output over {} frames", ARM_INTERP_FRAMES);
            }
        }

        if (arm_override_state_ == ArmOverrideState::BLEND_TO_VLA) {
            double t     = std::min(1.0, static_cast<double>(arm_override_frame_) /
                                         static_cast<double>(ARM_INTERP_FRAMES));
            double alpha = t * t * (3.0 - 2.0 * t);
            for (int k = 0; k < G1_23DOF_ARM_JOINT_COUNT; k++) {
                lowcmd->msg_.motor_cmd()[ARM_JOINT_INDICES[k]].q() = static_cast<float>(
                    arm_override_interp_start_[k] * (1.0 - alpha) + ARM_VLA_TARGETS[k] * alpha);
            }
            arm_override_frame_++;
            if (arm_override_frame_ >= ARM_INTERP_FRAMES) {
                arm_override_state_ = ArmOverrideState::HOLD_VLA;
                spdlog::info("Arm override: holding VLA ready pose. Press down-arrow to release.");
            }
        } else if (arm_override_state_ == ArmOverrideState::HOLD_VLA) {
            for (int k = 0; k < G1_23DOF_ARM_JOINT_COUNT; k++) {
                lowcmd->msg_.motor_cmd()[ARM_JOINT_INDICES[k]].q() = static_cast<float>(
                    ARM_VLA_TARGETS[k]);
            }
        } else if (arm_override_state_ == ArmOverrideState::BLEND_TO_POLICY) {
            // Blend from captured start toward the policy output (already written to lowcmd).
            double t     = std::min(1.0, static_cast<double>(arm_override_frame_) /
                                         static_cast<double>(ARM_INTERP_FRAMES));
            double alpha = t * t * (3.0 - 2.0 * t);
            for (int k = 0; k < G1_23DOF_ARM_JOINT_COUNT; k++) {
                double policy_q = static_cast<double>(
                    lowcmd->msg_.motor_cmd()[ARM_JOINT_INDICES[k]].q());
                lowcmd->msg_.motor_cmd()[ARM_JOINT_INDICES[k]].q() = static_cast<float>(
                    arm_override_interp_start_[k] * (1.0 - alpha) + policy_q * alpha);
            }
            arm_override_frame_++;
            if (arm_override_frame_ >= ARM_INTERP_FRAMES) {
                arm_override_state_ = ArmOverrideState::OFF;
                spdlog::info("Arm override: fully released. Policy controls arms.");
            }
        }

        // Only publish to rt/lowcmd when this controller is the active one.
        if (controller_switch_ && !controller_switch_->is_active()) return;
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
    static std::shared_ptr<isaaclab::KeyboardInput> keyboard_input;
    // Controller arbitration gate. Initialised in main() after ChannelFactory::Init().
    static std::unique_ptr<ControllerSwitchGate> controller_switch_;
    // Upper-body activation interpolation state (shared across all FSM states).
    static constexpr int G1_TOTAL_JOINTS      = 29;
    static constexpr int G1_UPPER_BODY_START  = 12;  ///< First waist/arm joint index.
    static constexpr int ACTIVATION_INTERP_FRAMES = 1000;  ///< ~1 s at 1 kHz.
    static bool activation_interp_active_;
    static int  activation_interp_frame_;
    static std::array<double, G1_TOTAL_JOINTS> activation_interp_start_;
    static constexpr int LOWER_BODY_INTERP_FRAMES = 0;  ///< ~0.1 s at 1 kHz.
    static bool lower_body_interp_active_;
    static int  lower_body_interp_frame_;
    /// Set to true by a state's exit() to suppress the next enter() blend.
    static bool skip_next_enter_interp_;

    // Arm override (VLA ready pose): up-arrow blends arms to fixed pose, down-arrow blends back.
    static constexpr int ARM_INTERP_FRAMES = 500;  ///< ~0.5 s at 1 kHz.
    static constexpr int G1_23DOF_ARM_JOINT_COUNT = 10;
    // Hardware motor indices for the 10 arm joints (left then right).
    static constexpr std::array<int, G1_23DOF_ARM_JOINT_COUNT> ARM_JOINT_INDICES = {
        15, 16, 17, 18, 19,   // left:  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
        22, 23, 24, 25, 26    // right: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
    };
    // Target positions for the VLA ready pose.
    static constexpr std::array<double, G1_23DOF_ARM_JOINT_COUNT> ARM_VLA_TARGETS = {
        -0.55,  0.25,  0.0,  -0.25,  0.0,   // left arm
        -0.55, -0.25,  0.0,  -0.25,  0.0    // right arm
    };

    enum class ArmOverrideState { OFF, BLEND_TO_VLA, HOLD_VLA, BLEND_TO_POLICY };
    static ArmOverrideState arm_override_state_;
    static int  arm_override_frame_;
    static std::array<double, G1_23DOF_ARM_JOINT_COUNT> arm_override_interp_start_;
};