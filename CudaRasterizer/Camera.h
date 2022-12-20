#pragma once
#include <glm/glm.hpp>
#include "glm/gtx/quaternion.hpp"

namespace Eu {
	class Camera final
	{
	public:
		Camera();
		const glm::mat4& GetViewProjectionMatrix() const;
		const glm::mat4& GetProjectionMatrix() const;
		glm::quat RotateDegrees(float angleRadians, const glm::vec3& axis);

		void CalculateProjectionMatrix(float fov, float aspectRatio = 1.77f);
		void CalcViewMatrix(glm::mat4x4 view, glm::vec3 pos);
		glm::mat4x4& GetONB();

		glm::vec3 getForward() const;

		glm::vec3 getLeft() const;

		glm::vec3 getUp() const;

		void RotateYaw(float angleRadians);
		void RotatePitch(float angleRadians);
		void RotateRoll(float angleRadians);

		void moveForward(float movement);

		void moveLeft(float movement);

		void moveUp(float movement);
private:
		void CalculateInverseONB();
private:
		float m_FarPlane = 1000.f;
		float m_NearPlane = 0.1f;

		glm::vec3 m_Position{};
		glm::quat m_CameraQuaternion{};

		glm::quat m_CameraQuaternionYaw{};
		glm::quat m_CameraQuaternionPitch{};

		glm::quat m_CameraQuaternionRoll{};

		glm::mat4 m_ViewProjMatrix{};
		glm::mat4 m_Proj{};
		glm::mat4 m_View{};


		//Camera Frustum
		//void CreateFrustumForCamera(float aspectRatio, float zNearPlane = 0.f, float zFarPlane = 1000.f);
		//Frustum m_CurrentFrustum;



	};
	//FRUSTUM CULLING
	//struct Plan
	//{ // unit vector
	//	Plan(glm::vec3 normal, float distance) : m_Normal{normal}, m_DistanceFromOrigin{distance}
	//	{

	//	}
	//	float m_DistanceFromOrigin = 0.f;
	//	glm::vec3 m_Normal = { 0.f, 1.f, 0.f };

	//};
	//struct Frustum
	//{
	//	Plan topFace;
	//	Plan bottomFace;

	//	Plan rightFace;
	//	Plan leftFace;

	//	Plan farFace;
	//	Plan nearFace;
	//};

}
