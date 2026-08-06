// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

/**
 * Extended joystick DSL with keyboard input support.
 *
 * Supports both joystick button names (A, B, X, Y, LT, RT, etc.) and
 * keyboard key names (key_1, key_space, key_f1, etc.)
 *
 * Keyboard key syntax:
 * - "key_1.on_pressed"     # Key 1 is just pressed
 * - "key_space"            # Spacebar is pressed
 * - "key_f1.on_pressed"    # F1 is just pressed
 * - "key_enter"            # Enter is pressed
 *
 * Mixed expressions:
 * - "LT + key_1.on_pressed"  # LT held + key 1 just pressed
 * - "key_1 | key_2"          # Key 1 or key 2 pressed
 */

#pragma once

#include "unitree_joystick_dsl.hpp"
#include "keyboard_input.h"

namespace unitree::common::dsl
{

    /**
     * @brief Extended input context that supports both joystick and keyboard
     */
    struct InputContext
    {
        const UnitreeJoystick *joystick = nullptr;
        isaaclab::KeyboardInput *keyboard = nullptr;
    };

    /**
     * @brief Check if a key name is a keyboard key
     */
    inline bool IsKeyboardKeyName(const std::string &name)
    {
        std::string lower_name = name;
        for (auto &c : lower_name)
            c = (char)tolower((unsigned char)c);

        // If starts with key_, it's a keyboard key
        if (lower_name.rfind("key_", 0) == 0)
        {
            return true;
        }

        return false;
    }

    /**
     * @brief Get KeyBase from either joystick or keyboard based on key name
     */
    inline const KeyBase *GetKeyExtended(const InputContext &ctx, const std::string &name)
    {
        std::string lower_name = name;
        for (auto &c : lower_name)
            c = (char)tolower((unsigned char)c);

        // Check if it's a keyboard key
        if (IsKeyboardKeyName(lower_name))
        {
            if (ctx.keyboard == nullptr)
            {
                throw std::runtime_error("Keyboard input not available for key: " + name);
            }
            const KeyBase *kb = isaaclab::GetKeyboardKey(*ctx.keyboard, lower_name);
            if (kb == nullptr)
            {
                throw std::runtime_error("Unknown keyboard key: " + name);
            }
            return kb;
        }

        // Otherwise, it's a joystick key
        if (ctx.joystick == nullptr)
        {
            throw std::runtime_error("Joystick not available for key: " + name);
        }
        return &GetKey(*ctx.joystick, name);
    }

    /**
     * @brief Compile AST to executable predicate that works with InputContext
     */
    inline std::function<bool(const InputContext &)> CompileExtended(const Node &n)
    {
        switch (n.kind)
        {
        case Node::kAtom:
        {
            Atom a = n.atom;
            return [a](const InputContext &ctx) -> bool
            {
                const KeyBase *kb = GetKeyExtended(ctx, a.name);
                switch (a.field)
                {
                case Field::kPressed:
                    return kb->pressed;
                case Field::kOnPressed:
                    return kb->on_pressed;
                case Field::kOnReleased:
                    return kb->on_released;
                case Field::kHoldTimeGE:
                    return kb->pressed && (kb->pressed_time >= a.hold_seconds);
                }
                return false;
            };
        }
        case Node::kNot:
        {
            auto child = CompileExtended(*n.lhs);
            return [child](const InputContext &ctx)
            { return !child(ctx); };
        }
        case Node::kAnd:
        {
            auto l = CompileExtended(*n.lhs);
            auto r = CompileExtended(*n.rhs);
            return [l, r](const InputContext &ctx)
            { return l(ctx) && r(ctx); };
        }
        case Node::kOr:
        {
            auto l = CompileExtended(*n.lhs);
            auto r = CompileExtended(*n.rhs);
            return [l, r](const InputContext &ctx)
            { return l(ctx) || r(ctx); };
        }
        }
        throw std::runtime_error("Invalid node kind");
    }

    /**
     * @brief Parse and compile expression with keyboard support
     */
    inline std::function<bool(const InputContext &)> CompileExpressionExtended(const std::string &expr)
    {
        Parser p(expr);
        auto ast = p.Parse();
        return CompileExtended(*ast);
    }

} // namespace unitree::common::dsl
