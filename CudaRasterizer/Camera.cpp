#include "Camera.h"
#include "Input.h"
#include "KeyMouseCodes.h"
#include <iostream>





Camera::Camera(float fov)
{
	m_ViewProjMatrix = glm::mat4(1.0);
	m_CameraQuaternionPitch = glm::angleAxis(glm::radians(0.f), glm::vec3(1, 0, 0));
	m_CameraQuaternionYaw = glm::angleAxis(glm::radians(0.f), glm::vec3(0, 1, 0));
	m_CameraQuaternionRoll = glm::angleAxis(glm::radians(0.f), glm::vec3(0, 0, 1));
	
	m_CameraQuaternion= glm::qua<float, glm::defaultp>(static_cast<float>(1), static_cast<float>(0), static_cast<float>(0), static_cast<float>(0));
	CalculateProjectionMatrix(fov, 800.f/ 800.f);
}
const glm::mat4& Camera::GetViewProjectionMatrix() const
{
	return m_ViewProjMatrix;
}

const glm::mat4& Camera::GetProjectionMatrix() const
{
	return m_Proj;
}





glm::quat Camera::RotateDegrees(float angleRadians, const glm::vec3& axis)
{
	float radians = glm::radians(angleRadians);
	glm::quat q = glm::angleAxis(radians, axis);
	return q;
}

void Camera::CalculateInverseONB()
{
	m_ViewProjMatrix = m_Proj * (m_View);

}
void Camera::CalculateProjectionMatrix(float fov, float aspectRatio)
{

	m_Proj = glm::perspective(glm::radians(fov / 2.f), aspectRatio, m_NearPlane, m_FarPlane);
	CalculateInverseONB();
}


void Camera::CalcViewMatrix(glm::mat4x4 view, glm::vec3 pos)
{
	m_CameraQuaternion = m_CameraQuaternionYaw * m_CameraQuaternionPitch;
	m_CameraQuaternion = glm::quat(m_CameraQuaternion.x, m_CameraQuaternion.y, m_CameraQuaternion.z, m_CameraQuaternion.w);
	m_View = glm::mat4_cast((m_CameraQuaternion));
	m_View = glm::translate(m_View, pos);
	CalculateInverseONB();
}



void Camera::UpdateCamera()
{

	glm::vec3 upVec = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3 forwardVec = getForward();
	glm::vec3 rightVec = getLeft();


	upVec = glm::cross(forwardVec, rightVec);

	glm::vec3 newPos = m_Position;


	int multiplier = 1;
	if (Input::IsKeyPressed(KEY_LEFT_ALT))
		multiplier++;
	if (Input::IsKeyPressed(KEY_D)) {
		newPos -= rightVec * (m_CameraMovementSpeed * multiplier);
		//EU_CORE_INFO("NEW POS: X {0}, Y {1}, Z {2}", newPos.x, newPos.y, newPos.z);
		m_UpdateNeeded = true;
	}
	if (Input::IsKeyPressed(KEY_A)) {
		newPos += rightVec * ((m_CameraMovementSpeed * multiplier));
		//EU_CORE_INFO("NEW POS: X {0}, Y {1}, Z {2}", newPos.x, newPos.y, newPos.z);

		m_UpdateNeeded = true;
	}
	if (Input::IsKeyPressed(KEY_W)) {
		newPos += forwardVec * ((m_CameraMovementSpeed * multiplier));
		//EU_CORE_INFO("NEW POS: X {0}, Y {1}, Z {2}", newPos.x, newPos.y, newPos.z);

		m_UpdateNeeded = true;
	}

	if (Input::IsKeyPressed(KEY_S)) {
		newPos -= forwardVec * ((m_CameraMovementSpeed * multiplier));
		//EU_CORE_INFO("NEW POS: X {0}, Y {1}, Z {2}", newPos.x, newPos.y, newPos.z);

		m_UpdateNeeded = true;
	}
	if (Input::IsKeyPressed(KEY_C)) {
		newPos += upVec * ((m_CameraMovementSpeed * multiplier));
		//EU_CORE_INFO("NEW POS: X {0}, Y {1}, Z {2}", newPos.x, newPos.y, newPos.z);

		m_UpdateNeeded = true;
	}
	if (Input::IsKeyPressed(KEY_SPACE)) {
		newPos -= upVec * ((m_CameraMovementSpeed * multiplier));
		//EU_CORE_INFO("NEW POS: X {0}, Y {1}, Z {2}", newPos.x, newPos.y, newPos.z);

		m_UpdateNeeded = true;

	}
	m_Position = newPos;
	//Rotation
	if (Input::IsMouseButtonPressed(MOUSE_BUTTON_2)) { //Remove make button configurable
		int x = Input::GetMouseX();
		int y = Input::GetMouseY();

		if (m_IsFirstUpdate)
		{
			m_OldScreenPos.x = x;
			m_OldScreenPos.y = y;

			m_IsFirstUpdate = false;
			return;
		}
		m_ScreenPosOffset = { x - m_OldScreenPos.x, y - m_OldScreenPos.y };

		m_OldScreenPos.x = x;
		m_OldScreenPos.y = y;

		//TODO: Use time
		m_ScreenPosOffset = m_ScreenPosOffset * (m_sensitivity);


		m_CameraRot.y += m_ScreenPosOffset.x; //yaw rotate x
		m_CameraRot.x += m_ScreenPosOffset.y; //pitch rotate y
		m_CameraRot.z = 0;



		if (m_CameraRot.x > 89.0f)
			m_CameraRot.x = 89.0f;
		if (m_CameraRot.x < -89.0f)
			m_CameraRot.x = -89.0f;

		RotateYaw(m_ScreenPosOffset.x);
		RotatePitch(m_ScreenPosOffset.y);

		m_UpdateNeeded = true;

	}
	int x = Input::GetMouseX();
	int y = Input::GetMouseY();
	m_OldScreenPos.x = x;
	m_OldScreenPos.y = y;


	if (m_UpdateNeeded) {

		auto worldPos = m_Position;
		auto look = getForward();
		std::cout << worldPos.x << " " << worldPos.y << " " << worldPos.z << std::endl;

		CalcViewMatrix((glm::lookAt(worldPos, worldPos + look, glm::vec3{ 0,1,0 })), worldPos);
		m_UpdateNeeded = false;
	}

}

glm::vec3 Camera::getForward() const
{
	return glm::conjugate(m_CameraQuaternion) * glm::vec3(0.0f, 0.0f, -1.0f);
}

glm::vec3 Camera::getLeft() const
{
	return glm::conjugate(m_CameraQuaternion) * glm::vec3(-1.0, 0.0f, 0.0f);
}

glm::vec3 Camera::getUp() const
{
	return glm::conjugate(m_CameraQuaternion) * glm::vec3(0.0f, -1.0f, 0.0f);
}

void Camera::RotateYaw(float angleRadians)
{
	float radians = glm::radians(angleRadians);
	m_CameraQuaternionYaw *=  RotateDegrees(radians, glm::vec3(0.0f, 1.0f, 0.0f));
}

void Camera::RotatePitch(float angleRadians)
{
	float radians = glm::radians(angleRadians);
	m_CameraQuaternionPitch *= RotateDegrees(radians, glm::vec3(0.0f, 0.0f, -1.0f));
}

void Camera::RotateRoll(float angleRadians)
{
	float radians = glm::radians(angleRadians);
	m_CameraQuaternionRoll *= RotateDegrees(radians, glm::vec3(0.0f, 1.0f, 0.0f));
}

void Camera::moveForward(float movement)
{
	m_Position += getForward() * movement;
}

void Camera::moveLeft(float movement)
{
	m_Position += getLeft() * movement;
}

void Camera::moveUp(float movement)
{
	m_Position += getUp() * movement;
}





