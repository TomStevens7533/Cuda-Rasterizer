#pragma once

#include <glfw3.h>
#include <vec2.hpp> // glm::vec3



class Input
{
public:
	inline static bool IsKeyPressed(int keycode);
	inline static bool IsMouseButtonPressed(int button);
	inline static glm::vec2 GetMousePosition();
	inline static float GetMouseX();
	inline static float GetMouseY();
	inline static glm::vec2 GetMouseDrag(int button);
};

