// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/ros2/Pose2D_.hpp>
#include <mutex>
#include <thread>
#include <spdlog/spdlog.h>
#include <eigen3/Eigen/Dense>

namespace isaaclab
{

    /**
     * @brief GoalPose data structure matching geometry_msgs/msg/Pose2D
     */
    struct GoalPoseData
    {
        float x = 0.0f;
        float y = 0.0f;
        float theta = 0.0f;

        // Flag to indicate if goal pose has been received
        bool is_valid = false;
    };

    /**
     * @brief Subscription class for receiving GoalPose data from DDS topic
     *
     * Subscribes to a DDS topic of type geometry_msgs/msg/Pose2D and provides
     * easy access to the goal pose data.
     */
    class GoalPoseSubscription
    {
    public:
        using MsgType = geometry_msgs::msg::dds_::Pose2D_;
        using SharedPtr = std::shared_ptr<GoalPoseSubscription>;

        explicit GoalPoseSubscription(const std::string &topic = "rt/goal_pose")
            : topic_(topic)
        {
            last_update_time_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(timeout_ms_);

            spdlog::info("Creating GoalPoseSubscription on topic: '{}'", topic);

            sub_ = std::make_shared<unitree::robot::ChannelSubscriber<MsgType>>(topic);
            sub_->InitChannel([this](const void *msg)
            {
                last_update_time_ = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> lock(mutex_);
                msg_ = *(const MsgType*)msg;
                
                // Update the convenient data structure
                data_.x = msg_.x();
                data_.y = msg_.y();
                data_.theta = msg_.theta();
                data_.is_valid = true;
                
                // spdlog::debug("GoalPose received: x={:.3f}, y={:.3f}, theta={:.3f}", data_.x, data_.y, data_.theta); 
            });
        }

        void set_timeout_ms(uint32_t timeout_ms) { timeout_ms_ = timeout_ms; }

        bool isTimeout() const
        {
            auto now = std::chrono::steady_clock::now();
            auto elasped_time = now - last_update_time_;
            return elasped_time > std::chrono::milliseconds(timeout_ms_);
        }

        void wait_for_connection()
        {
            auto t0 = std::chrono::steady_clock::now();
            bool warn_info = false;
            while (isTimeout())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (!warn_info && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2))
                {
                    warn_info = true;
                    spdlog::warn("Waiting for connection {}", topic_);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (warn_info)
            {
                spdlog::info("Connected {}", topic_);
            }
        }

        /**
         * @brief Get the current goal pose data (thread-safe)
         */
        GoalPoseData getData() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return data_;
        }

        /**
         * @brief Get direct access to the goal pose data
         * @warning Not thread-safe, use with caution or with external locking
         */
        const GoalPoseData &data() const { return data_; }
        GoalPoseData &data() { return data_; }

        /**
         * @brief Get raw message access
         */
        const MsgType &msg() const { return msg_; }

        mutable std::mutex mutex_;

    private:
        std::string topic_;
        uint32_t timeout_ms_{1000};
        unitree::robot::ChannelSubscriberPtr<MsgType> sub_;
        std::chrono::steady_clock::time_point last_update_time_;

        MsgType msg_;
        GoalPoseData data_;
    };

} // namespace isaaclab
