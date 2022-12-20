#pragma once

#include <vec2.hpp> // glm::vec3



class Input
{
public:
	static bool IsKeyPressed(int keycode);
	static bool IsMouseButtonPressed(int button);
	static glm::vec2 GetMousePosition();
	static float GetMouseX();
	static float GetMouseY();
	static glm::vec2 GetMouseDrag(int button);
private:
	static glm::vec2 m_PreviousMousePosDrag;
};

