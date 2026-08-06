#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"
#include "State_PosePID.h"
#include "State_PoseTracking.h"
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/ros2/Pose2D_.hpp>
#include "goal_pose.h"
#include <thread>

#include <array>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;
std::shared_ptr<isaaclab::KeyboardInput> FSMState::keyboard_input = nullptr;
std::unique_ptr<ControllerSwitchGate> FSMState::controller_switch_ = nullptr;
bool FSMState::activation_interp_active_ = false;
int  FSMState::activation_interp_frame_  = 0;
std::array<double, FSMState::G1_TOTAL_JOINTS> FSMState::activation_interp_start_ = {};
bool FSMState::lower_body_interp_active_ = false;
int  FSMState::lower_body_interp_frame_  = 0;
bool FSMState::skip_next_enter_interp_   = false;
FSMState::ArmOverrideState FSMState::arm_override_state_ = FSMState::ArmOverrideState::OFF;
int  FSMState::arm_override_frame_ = 0;
std::array<double, FSMState::G1_23DOF_ARM_JOINT_COUNT> FSMState::arm_override_interp_start_ = {};

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-23dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    // Controller arbitration: this process is controller ID=0, active by default.
    FSMState::controller_switch_ = std::make_unique<ControllerSwitchGate>(1, true);

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 4; // 23dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize keyboard input if enabled in config (MUST be before FSM initialization)
    bool enable_keyboard = false;
    try
    {
        enable_keyboard = param::config["FSM"]["enable_keyboard"].as<bool>();
    }
    catch (...)
    {
    }

    std::shared_ptr<std::thread> kb_pub_thread;
    if (enable_keyboard)
    {
        spdlog::info("Keyboard input enabled for FSM transitions");
        FSMState::keyboard_input = std::make_shared<isaaclab::KeyboardInput>();

        // Publish local terminal keypresses to rt/key_press so KeyboardInput picks them up
        auto kb_pub = std::make_shared<unitree::robot::ChannelPublisher<std_msgs::msg::dds_::String_>>("rt/key_press");
        kb_pub->InitChannel();
        
        // Goal pose publisher for w/a/s/d/r keys
        auto goal_pose_pub = std::make_shared<unitree::robot::ChannelPublisher<geometry_msgs::msg::dds_::Pose2D_>>("rt/goal_pose");
        goal_pose_pub->InitChannel();
        
        // Wait for DDS discovery (allow subscribers to find this publisher)
        usleep(100000); // 100ms
        
        // Initialize goal pose to zero
        geometry_msgs::msg::dds_::Pose2D_ goal_pose;
        goal_pose.x() = 0.0f;
        goal_pose.y() = 0.0f;
        goal_pose.theta() = 0.0f;
        goal_pose_pub->Write(goal_pose);
        spdlog::info("Goal pose initialized to (0, 0, 0). Use w/a/s/d keys to control, 'r' to reset");
        
        auto local_kb = std::make_shared<Keyboard>();
        kb_pub_thread = std::make_shared<std::thread>([local_kb, kb_pub, goal_pose_pub, goal_pose]() mutable
        {
            while (true)
            {
                local_kb->update();
                if (local_kb->on_pressed)
                {
                    std::string key = local_kb->key();
                    if (!key.empty())
                    {
                        std_msgs::msg::dds_::String_ msg;
                        msg.data() = key;
                        kb_pub->Write(msg);
                        spdlog::debug("Local keyboard: published key '{}'", key);
                        
                        // Handle goal pose control keys
                        bool pose_changed = false;
                        if (key == "w") {
                            goal_pose.x() += 0.1f;
                            pose_changed = true;
                            spdlog::info("Goal pose X increased to {:.2f}", goal_pose.x());
                        }
                        else if (key == "s") {
                            goal_pose.x() -= 0.1f;
                            pose_changed = true;
                            spdlog::info("Goal pose X decreased to {:.2f}", goal_pose.x());
                        }
                        else if (key == "a") {
                            goal_pose.y() += 0.1f;
                            pose_changed = true;
                            spdlog::info("Goal pose Y increased to {:.2f}", goal_pose.y());
                        }
                        else if (key == "d") {
                            goal_pose.y() -= 0.1f;
                            pose_changed = true;
                            spdlog::info("Goal pose Y decreased to {:.2f}", goal_pose.y());
                        }
                        else if (key == "q") {
                            goal_pose.theta() += 0.1f;
                            pose_changed = true;
                            spdlog::info("Goal pose theta increased to {:.2f}", goal_pose.theta());
                        }
                        else if (key == "e") {
                            goal_pose.theta() -= 0.1f;
                            pose_changed = true;
                            spdlog::info("Goal pose theta decreased to {:.2f}", goal_pose.theta());
                        }
                        else if (key == "r") {
                            goal_pose.x() = 0.0f;
                            goal_pose.y() = 0.0f;
                            goal_pose.theta() = 0.0f;
                            pose_changed = true;
                            spdlog::info("Goal pose reset to (0, 0, 0)");
                        }
                        
                        // Publish updated goal pose
                        if (pose_changed) {
                            goal_pose_pub->Write(goal_pose);
                            spdlog::debug("Published goal pose to rt/goal_pose: x={:.2f}, y={:.2f}", goal_pose.x(), goal_pose.y());
                        }
                    }
                }
                usleep(10000); // 10 ms
            }
        });
    }
    else
    {
        spdlog::warn("Keyboard input disabled but keyboard transitions may be present in config");
    }

    // Initialize FSM (keyboard_input must be initialized before this)
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

