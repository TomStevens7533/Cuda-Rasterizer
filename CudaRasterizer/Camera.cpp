#include "Camera.h"
#include <ostream>
#include <iostream>
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

#include <Europa/KeyMouseCodes.h>
#include "glm/gtx/quaternion.hpp"
#include "../Log.h"


namespace Eu {
	Camera::Camera()
	{
		m_ViewProjMatrix = glm::mat4(1.0);
		m_CameraQuaternionPitch = glm::angleAxis(glm::radians(0.f), glm::vec3(1, 0, 0));
		m_CameraQuaternionYaw = glm::angleAxis(glm::radians(0.f), glm::vec3(0, 1, 0));
		m_CameraQuaternionRoll = glm::angleAxis(glm::radians(0.f), glm::vec3(0, 0, 1));
		
		m_CameraQuaternion= glm::qua<float, glm::defaultp>(static_cast<float>(1), static_cast<float>(0), static_cast<float>(0), static_cast<float>(0));

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
	
		m_ViewProjMatrix = m_Proj * m_View;


	}
	void Camera::CalculateProjectionMatrix(float fov, float aspectRatio)
	{

		m_Proj = glm::perspective(glm::radians(fov / 2.f), aspectRatio, m_NearPlane, m_FarPlane);
		CalculateInverseONB();
	}


	void Camera::CalcViewMatrix(glm::mat4x4 view, glm::vec3 pos)
	{
		m_CameraQuaternion =  m_CameraQuaternionYaw * m_CameraQuaternionPitch;
		m_CameraQuaternion = glm::quat(m_CameraQuaternion.x, m_CameraQuaternion.y, m_CameraQuaternion.z, m_CameraQuaternion.w);
		m_View = glm::mat4_cast( ( m_CameraQuaternion));
		m_View = glm::translate(m_View, -pos);
		CalculateInverseONB();
	}

	glm::mat4x4& Camera::GetONB()
	{
		return m_ViewProjMatrix;
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
		return glm::conjugate(m_CameraQuaternion) * glm::vec3(0.0f, 1.0f, 0.0f);
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

	//frustum culling
	//void Camera::CreateFrustumForCamera(float aspectRatio, float zNearPlane = 0.f, float zFarPlane = 1000.f)
	//{
	//	const float halfVSide = zFarPlane * tanf(m_FOV * .5f);
	//	const float halfHSide = halfVSide * m_AspectRatio;
	//	const glm::vec3 frontMultFar = zFarPlane * m_ForwardVector;

	//	m_CurrentFrustum.nearFace = Plan{ glm::vec3{m_CameraPos + zNearPlane * m_ForwardVector}, m_ForwardVector };

	//	m_CurrentFrustum.farFace = Plan{ m_ForwardVector + frontMultFar, -m_ForwardVector };

	//	m_CurrentFrustum.rightFace = Plan{ m_CameraPos,
	//							glm::cross(m_UpVector,frontMultFar + m_RightVector * halfHSide) };

	//	m_CurrentFrustum.leftFace = Plan{ m_CameraPos,
	//							glm::cross(frontMultFar - m_RightVector * halfHSide, m_UpVector) };

	//	m_CurrentFrustum.topFace = Plan{ m_CameraPos,
	//							glm::cross(m_RightVector, frontMultFar - m_UpVector * halfVSide) };

	//	m_CurrentFrustum.bottomFace = Plan{ m_CameraPos,
	//							glm::cross(frontMultFar + m_UpVector * halfVSide, m_RightVector) };
	//}
}


