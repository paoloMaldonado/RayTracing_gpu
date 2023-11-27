#include "instance.cuh"
#include "utility/utils.cuh"

__device__
Instance::Instance(Shape* object_ptr) : object_ptr(object_ptr)	// the inv_matrix initialized here is the identity
{}

__device__
Instance::Instance(Shape* object_ptr, const Transform* transform) : 
	object_ptr(object_ptr), inv_matrix(transform)
{}

__device__
Instance::Instance(Shape* object_ptr, Material* material) :
	object_ptr(object_ptr)
{
	object_ptr->add_material(material);
}

__device__
Instance::Instance(Shape* object_ptr, const Transform* transform, Material* material) :
	object_ptr(object_ptr), inv_matrix(transform)
{
	object_ptr->add_material(material);
}

__device__
bool Instance::hitted_by(const Ray& ray, float& t, Ray& inv_ray, float& u, float& v) const
{
	//inv_ray = (*inv_matrix)(ray);

	if (object_ptr->hitted_by(ray, t, u, v))
	{
		// if no material is specified when the object is created, then the instance of this
		// object can have different materials
		return true;
	}
	return false;
}

__device__
normal3 Instance::compute_normal(const point3& p, const float& u, const float& v) const
{
	normal3 normal = object_ptr->compute_normal_at(p, u, v);		// compute the normal at the untransformed object
	//normal = (*inv_matrix)(normal);							    // transpose of the inverse matrix
	return normalize(normal);
}