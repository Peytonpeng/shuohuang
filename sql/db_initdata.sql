INSERT INTO "public"."tb_analysis_model"("model_id", "model_name", "create_user", "create_time", "param_config") VALUES ('1', '线性回归', 'admin', '2025-04-01 11:38:48', '[{"param_name": "lr_max_iter", "param_values": [100, 1000, 2000], "default_value": "100"}, {"param_name": "lr_C", "param_values": [0.1, 1, 10], "default_value": "1"}, {"param_name": "lr_solver", "param_values": ["liblinear", "lbfgs", "sage"], "default_value": "sage"}]');
INSERT INTO "public"."tb_analysis_model"("model_id", "model_name", "create_user", "create_time", "param_config") VALUES ('5', 'K-均值聚类', 'admin', '2025-04-01 11:38:48', '[{
	"param_name": "kmeans_n_clusters_param",
	"param_values": [5, 8, 10],
	"default_value": 5
}]');
INSERT INTO "public"."tb_analysis_model"("model_id", "model_name", "create_user", "create_time", "param_config") VALUES ('3', '随机森林', 'system', '2025-04-03 14:08:52', '[{
	"param_name": "n_estimators_param",
	"param_values": [50, 100, 200],
	"default_value": 50
}]');
INSERT INTO "public"."tb_analysis_model"("model_id", "model_name", "create_user", "create_time", "param_config") VALUES ('6', '深度神经网络', 'admin', '2025-04-01 11:38:48', '[{
		"param_name": "num_classes",
		"param_values": [3, 5, 10],
		"default_value": 5
	},
	{
		"param_name": "num_rounds",
		"param_values": [5, 10, 20],
		"default_value": 5
	},
	{
		"param_name": "epochs_per_round",
		"param_values": [100, 200, 300],
		"default_value": 100
	},
	{
		"param_name": "lr",
		"param_values": [0.01, 0.001, 0.0001],
		"default_value": 0.001
	}
]');
INSERT INTO "public"."tb_analysis_model"("model_id", "model_name", "create_user", "create_time", "param_config") VALUES ('4', '支持向量机', 'admin', '2025-04-01 11:38:48', '[{
		"param_name": "svc_C",
		"param_values": [0.1, 1, 10],
		"default_value": 1
	},
	{
		"param_name": "svc_kernel",
		"param_values": ["linear", "poly", "rbf", "sigmoid"],
		"default_value": "rbf"
	},
	{
		"param_name": "svc_gamma",
		"param_values": ["scale", "auto"],
		"default_value": "scale"
	}
]');
INSERT INTO "public"."tb_analysis_model"("model_id", "model_name", "create_user", "create_time", "param_config") VALUES ('7', 'RVFL回归器', 'admin', '2025-06-20 10:54:09', '[]');
INSERT INTO "public"."tb_analysis_model"("model_id", "model_name", "create_user", "create_time", "param_config") VALUES ('2', '逻辑回归', 'admin', '2025-04-01 11:38:48', '[{"default_value": 5, "param_name": "kmeans_n_clusters_param", "param_values": [5, 8, 10]}]');
