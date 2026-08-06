// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/ros2/String_.hpp>
#include <mutex>
#include <unordered_set>
#include <string>
#include <spdlog/spdlog.h>
#include <chrono>

namespace isaaclab
{

    using unitree::common::Button;
    using unitree::common::KeyBase;

    /**
     * @brief Keyboard input class that provides joystick-like button interface
     *
     * Tracks individual keys with pressed, on_pressed, on_released states
     * compatible with the joystick DSL transition system.
     * 
     * Subscribes to the rt/key_press DDS topic to receive key press events from MuJoCo simulation.
     */
    class KeyboardInput
    {
    public:
        using MsgType = std_msgs::msg::dds_::String_;

        KeyboardInput(const std::string &topic = "rt/key_press")
            : topic_(topic)
        {
            spdlog::info("Creating KeyboardInput subscriber on topic: '{}'", topic);

            sub_ = std::make_shared<unitree::robot::ChannelSubscriber<MsgType>>(topic);
            sub_->InitChannel([this](const void *msg)
                              {
                std::lock_guard<std::mutex> lock(mutex_);
                const MsgType* str_msg = (const MsgType*)msg;
                current_key_ = str_msg->data();
                
                spdlog::debug("Key press received: '{}'", current_key_); });
        }

        ~KeyboardInput() = default;

        /**
         * @brief Update all key states. Call once per control loop.
         */
        void update()
        {
            std::lock_guard<std::mutex> lock(mutex_);

            // Update all pre-defined keys based on current_key_
            updateAllKeys();

            // Reset current key after processing (single-frame detection)
            current_key_ = "";
        }

        // Common key accessors (pre-defined for convenience)
        Button<int> key_1, key_2, key_3, key_4, key_5, key_6, key_7, key_8, key_9, key_0;
        Button<int> key_t, key_y, key_u, key_i, key_o, key_p;
        Button<int> key_f, key_g, key_h, key_j, key_k, key_l;
        Button<int> key_z, key_x, key_c, key_v, key_b, key_n, key_m;
        Button<int> key_w, key_a, key_s, key_d, key_q, key_e;
        Button<int> key_space, key_enter, key_escape;
        Button<int> key_up, key_down, key_left, key_right;
        Button<int> key_f1, key_f2, key_f3, key_f4, key_f5, key_f6, key_f7, key_f8, key_f9, key_f10, key_f11, key_f12;

    private:
        std::string topic_;
        unitree::robot::ChannelSubscriberPtr<MsgType> sub_;
        mutable std::mutex mutex_;
        std::string current_key_;

        /**
         * @brief Update all pre-defined keys
         */
        void updateAllKeys()
        {
            // Numbers
            key_1(current_key_ == "1" ? 1 : 0);
            key_2(current_key_ == "2" ? 1 : 0);
            key_3(current_key_ == "3" ? 1 : 0);
            key_4(current_key_ == "4" ? 1 : 0);
            key_5(current_key_ == "5" ? 1 : 0);
            key_6(current_key_ == "6" ? 1 : 0);
            key_7(current_key_ == "7" ? 1 : 0);
            key_8(current_key_ == "8" ? 1 : 0);
            key_9(current_key_ == "9" ? 1 : 0);
            key_0(current_key_ == "0" ? 1 : 0);
            // Letters (excluding w, a, s, d, q, e)
            key_t(current_key_ == "t" ? 1 : 0);
            key_y(current_key_ == "y" ? 1 : 0);
            key_u(current_key_ == "u" ? 1 : 0);
            key_i(current_key_ == "i" ? 1 : 0);
            key_o(current_key_ == "o" ? 1 : 0);
            key_p(current_key_ == "p" ? 1 : 0);
            key_f(current_key_ == "f" ? 1 : 0);
            key_g(current_key_ == "g" ? 1 : 0);
            key_h(current_key_ == "h" ? 1 : 0);
            key_j(current_key_ == "j" ? 1 : 0);
            key_k(current_key_ == "k" ? 1 : 0);
            key_l(current_key_ == "l" ? 1 : 0);
            key_z(current_key_ == "z" ? 1 : 0);
            key_x(current_key_ == "x" ? 1 : 0);
            key_c(current_key_ == "c" ? 1 : 0);
            key_v(current_key_ == "v" ? 1 : 0);
            key_b(current_key_ == "b" ? 1 : 0);
            key_n(current_key_ == "n" ? 1 : 0);
            key_m(current_key_ == "m" ? 1 : 0);
            key_w(current_key_ == "w" ? 1 : 0);
            key_a(current_key_ == "a" ? 1 : 0);
            key_s(current_key_ == "s" ? 1 : 0);
            key_d(current_key_ == "d" ? 1 : 0);
            key_q(current_key_ == "q" ? 1 : 0);
            key_e(current_key_ == "e" ? 1 : 0);
            // Special keys
            key_space(current_key_ == "space" ? 1 : 0);
            key_enter(current_key_ == "enter" ? 1 : 0);
            key_escape(current_key_ == "escape" ? 1 : 0);
            key_up(current_key_ == "up" ? 1 : 0);
            key_down(current_key_ == "down" ? 1 : 0);
            key_left(current_key_ == "left" ? 1 : 0);
            key_right(current_key_ == "right" ? 1 : 0);
            // Function keys
            key_f1(current_key_ == "f1" ? 1 : 0);
            key_f2(current_key_ == "f2" ? 1 : 0);
            key_f3(current_key_ == "f3" ? 1 : 0);
            key_f4(current_key_ == "f4" ? 1 : 0);
            key_f5(current_key_ == "f5" ? 1 : 0);
            key_f6(current_key_ == "f6" ? 1 : 0);
            key_f7(current_key_ == "f7" ? 1 : 0);
            key_f8(current_key_ == "f8" ? 1 : 0);
            key_f9(current_key_ == "f9" ? 1 : 0);
            key_f10(current_key_ == "f10" ? 1 : 0);
            key_f11(current_key_ == "f11" ? 1 : 0);
            key_f12(current_key_ == "f12" ? 1 : 0);
        }
    };

    /**
     * @brief Get KeyBase from KeyboardInput by name
     */
    inline const KeyBase *GetKeyboardKey(KeyboardInput &kbd, const std::string &name)
    {
        std::string lower_name = name;
        for (auto &c : lower_name)
            c = (char)tolower((unsigned char)c);

        // Map key_X names to X
        std::string key_name = lower_name;
        if (lower_name.rfind("key_", 0) == 0)
        {
            key_name = lower_name.substr(4);
        }

        // Pre-defined keys
        if (key_name == "1")
            return &static_cast<const KeyBase &>(kbd.key_1);
        if (key_name == "2")
            return &static_cast<const KeyBase &>(kbd.key_2);
        if (key_name == "3")
            return &static_cast<const KeyBase &>(kbd.key_3);
        if (key_name == "4")
            return &static_cast<const KeyBase &>(kbd.key_4);
        if (key_name == "5")
            return &static_cast<const KeyBase &>(kbd.key_5);
        if (key_name == "6")
            return &static_cast<const KeyBase &>(kbd.key_6);
        if (key_name == "7")
            return &static_cast<const KeyBase &>(kbd.key_7);
        if (key_name == "8")
            return &static_cast<const KeyBase &>(kbd.key_8);
        if (key_name == "9")
            return &static_cast<const KeyBase &>(kbd.key_9);
        if (key_name == "0")
            return &static_cast<const KeyBase &>(kbd.key_0);
        if (key_name == "t")
            return &static_cast<const KeyBase &>(kbd.key_t);
        if (key_name == "y")
            return &static_cast<const KeyBase &>(kbd.key_y);
        if (key_name == "u")
            return &static_cast<const KeyBase &>(kbd.key_u);
        if (key_name == "i")
            return &static_cast<const KeyBase &>(kbd.key_i);
        if (key_name == "o")
            return &static_cast<const KeyBase &>(kbd.key_o);
        if (key_name == "p")
            return &static_cast<const KeyBase &>(kbd.key_p);
        if (key_name == "f")
            return &static_cast<const KeyBase &>(kbd.key_f);
        if (key_name == "g")
            return &static_cast<const KeyBase &>(kbd.key_g);
        if (key_name == "h")
            return &static_cast<const KeyBase &>(kbd.key_h);
        if (key_name == "j")
            return &static_cast<const KeyBase &>(kbd.key_j);
        if (key_name == "k")
            return &static_cast<const KeyBase &>(kbd.key_k);
        if (key_name == "l")
            return &static_cast<const KeyBase &>(kbd.key_l);
        if (key_name == "z")
            return &static_cast<const KeyBase &>(kbd.key_z);
        if (key_name == "x")
            return &static_cast<const KeyBase &>(kbd.key_x);
        if (key_name == "c")
            return &static_cast<const KeyBase &>(kbd.key_c);
        if (key_name == "v")
            return &static_cast<const KeyBase &>(kbd.key_v);
        if (key_name == "b")
            return &static_cast<const KeyBase &>(kbd.key_b);
        if (key_name == "n")
            return &static_cast<const KeyBase &>(kbd.key_n);
        if (key_name == "m")
            return &static_cast<const KeyBase &>(kbd.key_m);
        if (key_name == "w")
            return &static_cast<const KeyBase &>(kbd.key_w);
        if (key_name == "a")
            return &static_cast<const KeyBase &>(kbd.key_a);
        if (key_name == "s")
            return &static_cast<const KeyBase &>(kbd.key_s);
        if (key_name == "d")
            return &static_cast<const KeyBase &>(kbd.key_d);
        if (key_name == "q")
            return &static_cast<const KeyBase &>(kbd.key_q);
        if (key_name == "e")
            return &static_cast<const KeyBase &>(kbd.key_e);
        if (key_name == "space")
            return &static_cast<const KeyBase &>(kbd.key_space);
        if (key_name == "enter")
            return &static_cast<const KeyBase &>(kbd.key_enter);
        if (key_name == "escape" || key_name == "esc")
            return &static_cast<const KeyBase &>(kbd.key_escape);
        if (key_name == "up")
            return &static_cast<const KeyBase &>(kbd.key_up);
        if (key_name == "down")
            return &static_cast<const KeyBase &>(kbd.key_down);
        if (key_name == "left")
            return &static_cast<const KeyBase &>(kbd.key_left);
        if (key_name == "right")
            return &static_cast<const KeyBase &>(kbd.key_right);
        if (key_name == "f1")
            return &static_cast<const KeyBase &>(kbd.key_f1);
        if (key_name == "f2")
            return &static_cast<const KeyBase &>(kbd.key_f2);
        if (key_name == "f3")
            return &static_cast<const KeyBase &>(kbd.key_f3);
        if (key_name == "f4")
            return &static_cast<const KeyBase &>(kbd.key_f4);
        if (key_name == "f5")
            return &static_cast<const KeyBase &>(kbd.key_f5);
        if (key_name == "f6")
            return &static_cast<const KeyBase &>(kbd.key_f6);
        if (key_name == "f7")
            return &static_cast<const KeyBase &>(kbd.key_f7);
        if (key_name == "f8")
            return &static_cast<const KeyBase &>(kbd.key_f8);
        if (key_name == "f9")
            return &static_cast<const KeyBase &>(kbd.key_f9);
        if (key_name == "f10")
            return &static_cast<const KeyBase &>(kbd.key_f10);
        if (key_name == "f11")
            return &static_cast<const KeyBase &>(kbd.key_f11);
        if (key_name == "f12")
            return &static_cast<const KeyBase &>(kbd.key_f12);

        return nullptr;
    }

    /**
     * @brief Check if a key name is a keyboard key (starts with key_ or is a known keyboard key)
     */
    inline bool IsKeyboardKey(const std::string &name)
    {
        std::string lower_name = name;
        for (auto &c : lower_name)
            c = (char)tolower((unsigned char)c);

        // If starts with key_, it's definitely a keyboard key
        if (lower_name.rfind("key_", 0) == 0)
        {
            return true;
        }

        // Check for known keyboard-only keys
        // Note: w, a, s, d, q, e are excluded - reserved for velocity commands
        static const std::unordered_set<std::string> keyboard_keys = {
            "space", "enter", "escape", "esc",
            "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
            "r", "t", "y", "u", "i", "o", "p",
            "f", "g", "h", "j", "k", "l",
            "z", "x", "c", "v", "b", "n", "m"};

        return keyboard_keys.count(lower_name) > 0;
    }

} // namespace isaaclab
