#include "Input.h"


bool Input::IsKeyPressed(int keycode)
{
	return true;
}

bool Input::IsMouseButtonPressed(int button)
{
	return true;

}

glm::vec2 Input::GetMousePosition()
{
	return glm::vec2{ 0,0 };
}

float Input::GetMouseX()
{
	return 0.f;
}

float Input::GetMouseY()
{
	return 0.f;

}

glm::vec2 Input::GetMouseDrag(int button)
{
	return glm::vec2{ 0,0 };

}

