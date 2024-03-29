#pragma once
#include <glm.hpp>
#include "gtx/quaternion.hpp"

class Camera final
{
public:
	Camera(float fov);
	const glm::mat4& GetViewProjectionMatrix() const;
	const glm::mat4& GetProjectionMatrix() const;
	glm::quat RotateDegrees(float angleRadians, const glm::vec3& axis);

	void CalculateProjectionMatrix(float fov, float aspectRatio = 1.77f);
	void CalcViewMatrix(glm::mat4x4 view, glm::vec3 pos);

	void UpdateCamera();


	glm::vec3 getForward() const;
	glm::vec3 getLeft() const;
	glm::vec3 getUp() const;

	void CalculateInverseONB();

	glm::vec3 GetPosition();
private:
	void RotateYaw(float angleRadians);
	void RotatePitch(float angleRadians);
	void RotateRoll(float angleRadians);

	void moveForward(float movement);
	void moveLeft(float movement);
	void moveUp(float movement);
private:
	float m_FarPlane = 1000000.f;
	float m_NearPlane = 0.01f;

	glm::vec3 m_Position{0, 500,50 };
	glm::quat m_CameraQuaternion{};

	glm::quat m_CameraQuaternionYaw{};
	glm::quat m_CameraQuaternionPitch{};

	glm::quat m_CameraQuaternionRoll{};

	glm::mat4 m_ViewProjMatrix{};
	glm::mat4 m_Proj{};
	glm::mat4 m_View{};

	float m_FOV;
	bool m_UpdateNeeded{ true };

	bool m_IsFirstUpdate;
	glm::vec2 m_OldScreenPos;
	glm::vec2 m_ScreenPosOffset;
	glm::vec3 m_CameraRot{};




	float m_sensitivity = 150.f;
	//variables
	float m_CameraMovementSpeed{ 25.f };


};

