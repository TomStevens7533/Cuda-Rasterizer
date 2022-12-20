#include "Input.h"
#include "KeyMouseCodes.h"
#include "Application.h"
#include <glfw3.h>

glm::vec2 Input::m_PreviousMousePosDrag;

bool Input::IsKeyPressed(int keycode)
{
	auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetWindow());
	auto state = glfwGetKey(window, keycode);

	return state == PRESS || state == REPEAT;
}

bool Input::IsMouseButtonPressed(int button)
{
	auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetWindow());
	auto state = glfwGetMouseButton(window, button);

	return state == PRESS;

}

glm::vec2 Input::GetMousePosition()
{
	auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetWindow());
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	return { (float)xpos, (float)ypos };
}
float Input::GetMouseX()
{
	return GetMousePosition().x;
}

float Input::GetMouseY()
{
	return GetMousePosition().y;

}

glm::vec2 Input::GetMouseDrag(int button)
{
	if (IsMouseButtonPressed(button) && m_PreviousMousePosDrag != glm::vec2{ 0,0 }) {

		auto currentPair = GetMousePosition();
		glm::vec2 dragDifference = m_PreviousMousePosDrag - currentPair;
		m_PreviousMousePosDrag = currentPair;//Set previous mouse pos

		return dragDifference;
	}
	auto currentPair = GetMousePosition();
	m_PreviousMousePosDrag = currentPair;//Set previous mouse pos
	return { 0,0 };

}

