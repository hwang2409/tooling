#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn base64_vectors_decode_without_a_crate() {
        assert_eq!(decode_base64(""), Ok(Vec::new()));
        assert_eq!(decode_base64("SGVsbG8="), Ok(b"Hello".to_vec()));
        assert_eq!(decode_base64("AAEC"), Ok(vec![0, 1, 2]));
        assert!(decode_base64("A===").is_err());
    }

    #[test]
    fn slerp_uses_the_shortest_path() {
        let a = [0.0, 0.0, 0.0, 1.0];
        let b = [0.0, 0.0, (FRAC_PI_2 / 2.0).sin(), (FRAC_PI_2 / 2.0).cos()];
        let quarter = slerp(a, b, 0.25);
        let expected = (FRAC_PI_2 / 8.0).sin();
        assert!((quarter[2] - expected).abs() < 1.0e-5);
        assert!((quarter[3] - (FRAC_PI_2 / 8.0).cos()).abs() < 1.0e-5);
    }

    #[test]
    fn trs_is_translation_rotation_scale() {
        let transform = NodeTransform {
            translation: Vec3::new(2.0, 0.0, 0.0),
            rotation: [0.0, 0.0, (FRAC_PI_2 / 2.0).sin(), (FRAC_PI_2 / 2.0).cos()],
            scale: Vec3::new(3.0, 3.0, 3.0),
        }
        .matrix();
        let result = transform * Vec4::new(1.0, 0.0, 0.0, 1.0);
        assert!((result.x - 2.0).abs() < 1.0e-5);
        assert!((result.y - 3.0).abs() < 1.0e-5);
        assert_eq!(result.z, 0.0);
        assert_eq!(result.w, 1.0);
    }

    #[test]
    fn rejects_accessor_overrun() {
        let source = r#"{"asset":{"version":"2.0"},"buffers":[{"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAA","byteLength":12}],"bufferViews":[{"buffer":0,"byteLength":12}],"accessors":[{"bufferView":0,"componentType":5126,"count":2,"type":"VEC3"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}]}"#;
        assert!(GltfAsset::from_str(source, ".").is_err());
    }

    #[test]
    fn invalid_utf8_error_displays_the_byte_offset() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("invalid-utf8.gltf");
        std::fs::write(&path, [b'{', b'}', 0xff]).unwrap();
        let error = GltfAsset::load(&path).unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert_eq!(error.offset, Some(2));
        assert!(error.to_string().contains("byte 2:"));
    }

    #[test]
    fn vertex_semantics_reject_wrong_component_types() {
        let source = r#"{"asset":{"version":"2.0"},"buffers":[{"uri":"data:application/octet-stream;base64,AAECAw==","byteLength":3}],"bufferViews":[{"buffer":0,"byteLength":3}],"accessors":[{"bufferView":0,"componentType":5121,"count":1,"type":"VEC3"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}]}"#;
        let error = GltfAsset::from_str(source, ".").unwrap_err();
        assert!(error.message.contains("POSITION accessor 0"));
    }

    #[test]
    fn normalize_weights_accepts_the_exact_tolerance_boundary() {
        assert_eq!(normalize_weights([WEIGHT_TOLERANCE, 0.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.0]);
    }

    fn asset_with_nodes(nodes: Vec<GltfNode>) -> GltfAsset {
        GltfAsset {
            meshes: Vec::new(),
            nodes,
            scenes: Vec::new(),
            skins: Vec::new(),
            animations: Vec::new(),
            materials: Vec::new(),
            default_scene: 0,
        }
    }

    fn node(children: Vec<usize>, translation: Vec3) -> GltfNode {
        GltfNode {
            name: String::new(),
            children,
            mesh: None,
            skin: None,
            matrix: None,
            translation,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: Vec3::new(1.0, 1.0, 1.0),
        }
    }

    #[test]
    fn node_world_transforms_include_all_ancestors() {
        let asset = asset_with_nodes(vec![
            node(vec![1], Vec3::new(1.0, 0.0, 0.0)),
            node(vec![2], Vec3::new(0.0, 2.0, 0.0)),
            node(Vec::new(), Vec3::new(0.0, 0.0, 3.0)),
        ]);
        let worlds = asset.node_world_transforms(None, 0.0).unwrap();
        assert_eq!(worlds[2] * Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(1.0, 2.0, 3.0, 1.0));
    }

    #[test]
    fn node_world_transforms_allow_child_before_parent() {
        let asset = asset_with_nodes(vec![
            node(Vec::new(), Vec3::new(0.0, 3.0, 0.0)),
            node(vec![0], Vec3::new(2.0, 0.0, 0.0)),
        ]);
        let worlds = asset.node_world_transforms(None, 0.0).unwrap();
        assert_eq!(worlds[0] * Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(2.0, 3.0, 0.0, 1.0));
    }

    #[test]
    fn gltf_factors_are_linear_at_the_shader_boundary() {
        let material = GltfMaterial {
            name: String::new(),
            base_color_factor: Vec4::new(0.5, 0.5, 0.5, 1.0),
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            albedo_texture: None,
            normal_map_texture: None,
            alpha_mode: GltfAlphaMode::Opaque,
            alpha_cutoff: 0.5,
        };
        let (diffuse, specular, shininess, _) = material.blinn_phong_parameters();
        let uniforms = BlinnPhongUniforms::new_with_linear_colors(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Vec3::ZERO,
            diffuse,
            specular,
            shininess,
            Vec3::ZERO,
            DirectionalLight::new(Vec3::ZERO, Vec3::ZERO),
            PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
        );
        assert_eq!(uniforms.diffuse_color(), Vec3::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn alpha_modes_are_parsed_and_mask_is_rejected() {
        let source = r#"{"asset":{"version":"2.0"},"buffers":[],"bufferViews":[],"accessors":[],"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1,1,1,0]}},{"alphaMode":"BLEND","pbrMetallicRoughness":{"baseColorFactor":[1,1,1,0.5]}}]}"#;
        let asset = GltfAsset::from_str(source, ".").unwrap();
        assert_eq!(asset.materials[0].alpha_mode, GltfAlphaMode::Opaque);
        assert_eq!(asset.materials[1].alpha_mode, GltfAlphaMode::Blend);
        let mask = r#"{"asset":{"version":"2.0"},"buffers":[],"bufferViews":[],"accessors":[],"materials":[{"alphaMode":"MASK"}]}"#;
        assert!(GltfAsset::from_str(mask, ".")
            .unwrap_err()
            .message
            .contains("MASK unsupported"));
    }

    #[test]
    fn deep_node_hierarchy_is_iterative() {
        let mut nodes = Vec::with_capacity(10_000);
        for index in 0..10_000 {
            let children = if index + 1 < 10_000 {
                vec![index + 1]
            } else {
                Vec::new()
            };
            nodes.push(node(children, Vec3::new(1.0, 0.0, 0.0)));
        }
        let worlds = asset_with_nodes(nodes).node_world_transforms(None, 0.0).unwrap();
        assert_eq!(worlds[9_999] * Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(10_000.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn node_hierarchy_cycles_are_rejected() {
        let asset = asset_with_nodes(vec![node(vec![1], Vec3::ZERO), node(vec![0], Vec3::ZERO)]);
        assert!(asset.node_world_transforms(None, 0.0).is_err());
    }

    #[test]
    fn skins_the_hand_authored_arm_at_bind_and_rotated_pose() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/arm.gltf");
        let asset = GltfAsset::load(path).unwrap();
        let bind = asset.pose_mesh(0, 0, 0, Some(0), 0.0).unwrap();
        assert_eq!(bind.vertex(2).unwrap().position(), Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(
            bind.vertex(2).unwrap().normal(),
            Some(Vec3::new(0.0, 0.0, 1.0))
        );
        let posed = asset.pose_mesh(0, 0, 0, Some(0), 1.0).unwrap();
        let position = posed.vertex(2).unwrap().position();
        assert!((position.x - 0.0).abs() < 1.0e-5);
        assert!((position.y - 0.0).abs() < 1.0e-5);
        assert_eq!(
            posed.vertex(2).unwrap().normal(),
            Some(Vec3::new(0.0, 0.0, 1.0))
        );
    }

    #[test]
    fn skin_weights_are_normalized_with_zero_weight_fallback() {
        assert_eq!(
            normalize_weights([2.0, 1.0, 0.0, 0.0]),
            [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]
        );
        assert_eq!(normalize_weights([0.0; 4]), [0.0; 4]);
    }

    #[test]
    fn animation_sampling_covers_exact_between_and_step_times() {
        let linear = GltfAnimationSampler {
            input: vec![0.0, 1.0],
            output: vec![[0.0, 0.0, 0.0, 0.0], [2.0, 4.0, 6.0, 0.0]],
            output_components: 3,
            interpolation: Interpolation::Linear,
        };
        assert_eq!(linear.sample(0.0).unwrap(), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(linear.sample(1.0).unwrap(), [2.0, 4.0, 6.0, 0.0]);
        assert_eq!(linear.sample(0.5).unwrap(), [1.0, 2.0, 3.0, 0.0]);
        let step = GltfAnimationSampler {
            interpolation: Interpolation::Step,
            ..linear
        };
        assert_eq!(step.sample(0.5).unwrap(), [0.0, 0.0, 0.0, 0.0]);
    }
}
