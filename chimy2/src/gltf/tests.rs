#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;
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
            weights: None,
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
        let parameters = material.render_parameters();
        let lighting = make_gltf_lighting(
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            Mat4::IDENTITY,
            parameters,
            Vec3::ZERO,
            DirectionalLight::new(Vec3::ZERO, Vec3::ZERO),
            PointLight::new(Vec3::ZERO, Vec3::ZERO, 1.0, 0.0, 0.0),
        );
        let uniforms = CookTorranceUniforms::new_with_linear_base_color(
            lighting,
            parameters.base_color,
            parameters.metallic,
            parameters.roughness,
        );
        let varyings = CookTorranceVaryings {
            world_position: Vec3::ZERO,
            normal: Vec3::new(0.0, 0.0, 1.0),
            light_space_position: Vec4::new(0.0, 0.0, 0.0, 1.0),
        };
        // The production GGX draw sees the linear factor 0.5 once through
        // neutral ambient 0.1: 0.1 * 0.5 = 0.05 linear, or sRGB 63.
        assert_eq!(CookTorranceShader::shade(&varyings, &uniforms), 0xff3f3f3f);
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

    #[test]
    fn morph_blend_math_uses_production_position_and_normal_path() {
        let base = Mesh::new(
            vec![MeshVertex::new(
                Vec3::new(1.0, 2.0, 3.0),
                None,
                Some(Vec3::new(0.0, 0.0, 1.0)),
            )],
            Vec::new(),
        );
        let targets = vec![
            MorphTarget::new(
                vec![Vec3::new(1.0, 0.0, 0.0)],
                Some(vec![Vec3::new(1.0, 0.0, 0.0)]),
            )
            .unwrap(),
            MorphTarget::new(
                vec![Vec3::new(0.0, 2.0, 0.0)],
                Some(vec![Vec3::new(0.0, 1.0, 0.0)]),
            )
            .unwrap(),
        ];
        let blended = blend_morph_targets(&base, &targets, &[0.3, 0.7]).unwrap();
        assert_eq!(blended.vertex(0).unwrap().position(), Vec3::new(1.3, 3.4, 3.0));
        let expected_length = (0.3_f32 * 0.3 + 0.7 * 0.7 + 1.0).sqrt();
        let expected = Vec3::new(0.3, 0.7, 1.0) / expected_length;
        let normal = blended.vertex(0).unwrap().normal().unwrap();
        assert!((normal.x - expected.x).abs() < 1.0e-6);
        assert!((normal.y - expected.y).abs() < 1.0e-6);
        assert!((normal.z - expected.z).abs() < 1.0e-6);
    }

    #[test]
    fn morph_zero_and_one_weights_are_exact_identities() {
        let base = Mesh::new(
            vec![MeshVertex::new(Vec3::new(1.0, 2.0, 3.0), None, None)],
            Vec::new(),
        );
        let target = MorphTarget::new(vec![Vec3::new(4.0, 5.0, 6.0)], None).unwrap();
        let zero = blend_morph_targets(&base, std::slice::from_ref(&target), &[0.0]).unwrap();
        let one = blend_morph_targets(&base, std::slice::from_ref(&target), &[1.0]).unwrap();
        assert_eq!(zero, base);
        assert_eq!(one.vertex(0).unwrap().position(), Vec3::new(5.0, 7.0, 9.0));
    }

    fn base64(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut output = String::new();
        for chunk in bytes.chunks(3) {
            let first = chunk[0];
            let second = chunk.get(1).copied().unwrap_or(0);
            let third = chunk.get(2).copied().unwrap_or(0);
            output.push(ALPHABET[(first >> 2) as usize] as char);
            output.push(ALPHABET[((first & 3) << 4 | second >> 4) as usize] as char);
            output.push(if chunk.len() > 1 {
                ALPHABET[((second & 15) << 2 | third >> 6) as usize] as char
            } else {
                '='
            });
            output.push(if chunk.len() > 2 {
                ALPHABET[(third & 63) as usize] as char
            } else {
                '='
            });
        }
        output
    }

    fn morph_animation_json() -> String {
        let mut bytes = Vec::new();
        for values in [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ] as [[f32; 3]; 12] {
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        for value in [0.0_f32, 1.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for values in [[0.0_f32, 0.0, 0.0], [1.0, 1.0, 1.0]] {
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        let source = format!(
            r#"{{"asset":{{"version":"2.0"}},"buffers":[{{"uri":"data:application/octet-stream;base64,{}","byteLength":{}}}],"bufferViews":[{{"buffer":0,"byteLength":{}}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":0,"byteOffset":36,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":0,"byteOffset":72,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":0,"byteOffset":108,"componentType":5126,"count":2,"type":"SCALAR"}},{{"bufferView":0,"byteOffset":116,"componentType":5126,"count":2,"type":"VEC2"}}],"meshes":[{{"weights":[0.2,0.4],"primitives":[{{"attributes":{{"POSITION":0}},"targets":[{{"POSITION":1}},{{"POSITION":2}}]}}]}}],"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}],"scene":0,"animations":[{{"samplers":[{{"input":3,"output":4}}],"channels":[{{"sampler":0,"target":{{"node":0,"path":"weights"}}}}]}}]}}"#,
            base64(&bytes),
            bytes.len(),
            bytes.len(),
        );
        source
            .replace(
                "\"byteOffset\":108,\"componentType\":5126,\"count\":2,\"type\":\"SCALAR\"},{\"bufferView\":0,\"byteOffset\":116,\"componentType\":5126,\"count\":2,\"type\":\"VEC2\"",
                "\"byteOffset\":108,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},{\"bufferView\":0,\"byteOffset\":144,\"componentType\":5126,\"count\":2,\"type\":\"SCALAR\"},{\"bufferView\":0,\"byteOffset\":152,\"componentType\":5126,\"count\":6,\"type\":\"SCALAR\"",
            )
            .replace(
                "\"weights\":[0.2,0.4],\"primitives\":[{\"attributes\":{\"POSITION\":0},\"targets\":[{\"POSITION\":1},{\"POSITION\":2}]}]",
                "\"weights\":[0.2,0.4,0.1],\"primitives\":[{\"attributes\":{\"POSITION\":0},\"targets\":[{\"POSITION\":1},{\"POSITION\":2},{\"POSITION\":3}]}]",
            )
            .replace("\"input\":3,\"output\":4", "\"input\":4,\"output\":5")
    }

    #[test]
    fn gltf_morph_targets_and_weight_animation_follow_spec_layout() {
        let asset = GltfAsset::from_str(&morph_animation_json(), ".").unwrap();
        let primitive = &asset.meshes[0].primitives[0];
        assert_eq!(primitive.morph_targets.len(), 3);
        assert_eq!(primitive.morph_targets[0].position_deltas[0], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(primitive.morph_targets[1].position_deltas[0], Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(
            primitive.morph_targets[2].position_deltas[0],
            Vec3::new(0.0, 0.0, 1.0)
        );
        assert_eq!(primitive.morph_weights(), &[0.2, 0.4, 0.1]);
        assert_eq!(
            asset.sample_morph_weights(Some(0), 0.0, 0, 0, 0).unwrap(),
            [0.0, 0.0, 0.0]
        );
        assert_eq!(
            asset.sample_morph_weights(Some(0), 0.5, 0, 0, 0).unwrap(),
            [0.5, 0.5, 0.5]
        );
        assert_eq!(
            asset.sample_morph_weights(Some(0), 1.0, 0, 0, 0).unwrap(),
            [1.0, 1.0, 1.0]
        );
        let posed = asset
            .pose_mesh_with_weights(0, 0, 0, Some(0), 0.5, &[0.3, 0.7, 0.2])
            .unwrap();
        assert_eq!(posed.vertex(0).unwrap().position(), Vec3::new(0.3, 0.7, 0.2));
        let draws = asset.scene_draws(0, Some(0), 0.5).unwrap();
        assert_eq!(draws[0].mesh.vertex(0).unwrap().position(), Vec3::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn morph_setters_sanitize_immediately() {
        let mut asset = GltfAsset::from_str(&morph_animation_json(), ".").unwrap();
        let primitive = &mut asset.meshes[0].primitives[0];
        primitive
            .set_morph_weights(&[f32::NAN, f32::INFINITY, 2.0])
            .unwrap();
        assert_eq!(primitive.morph_weights(), &[0.0, 0.0, 1.0]);
        primitive.set_morph_weight(1, 2.0).unwrap();
        assert_eq!(primitive.morph_weights(), &[0.0, 1.0, 1.0]);
    }

    #[test]
    fn node_weights_override_mesh_defaults_until_animation_is_active() {
        let source = morph_animation_json().replace(
            "\"nodes\":[{\"mesh\":0}]",
            "\"nodes\":[{\"mesh\":0,\"weights\":[0.9,0.1,0.0]}]",
        );
        let asset = GltfAsset::from_str(&source, ".").unwrap();
        let draws = asset.scene_draws(0, None, 0.0).unwrap();
        assert_eq!(
            draws[0].mesh.vertex(0).unwrap().position(),
            Vec3::new(0.9, 0.1, 0.0)
        );
        let animated = asset.scene_draws(0, Some(0), 0.5).unwrap();
        assert_eq!(
            animated[0].mesh.vertex(0).unwrap().position(),
            Vec3::new(0.5, 0.5, 0.5)
        );
    }

    #[test]
    fn malformed_morph_inputs_return_errors_without_panicking() {
        let missing_accessor = morph_animation_json().replace("\"POSITION\":2", "\"POSITION\":99");
        assert!(GltfAsset::from_str(&missing_accessor, ".").is_err());
        let wrong_target_count =
            morph_animation_json().replace("\"weights\":[0.2,0.4,0.1]", "\"weights\":[0.2]");
        assert!(GltfAsset::from_str(&wrong_target_count, ".").is_err());
        let wrong_node_weights = morph_animation_json().replace(
            "\"nodes\":[{\"mesh\":0}]",
            "\"nodes\":[{\"mesh\":0,\"weights\":[0.9,0.1]}]",
        );
        assert!(GltfAsset::from_str(&wrong_node_weights, ".").is_err());
        let wrong_delta_count = morph_animation_json().replace(
            "\"byteOffset\":36,\"componentType\":5126,\"count\":3",
            "\"byteOffset\":36,\"componentType\":5126,\"count\":2",
        );
        assert!(GltfAsset::from_str(&wrong_delta_count, ".").is_err());
    }
}
