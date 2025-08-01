/*
 Navicat Premium Data Transfer

 Source Server         : 10.10.1.127
 Source Server Type    : PostgreSQL
 Source Server Version : 120014
 Source Host           : 10.10.1.127:15432
 Source Catalog        : shuohuang
 Source Schema         : public

 Target Server Type    : PostgreSQL
 Target Server Version : 120014
 File Encoding         : 65001

 Date: 15/07/2025 11:56:13
*/


-- ----------------------------
-- Table structure for tb_analysis_apply_check
-- ----------------------------
DROP TABLE IF EXISTS "public"."tb_analysis_apply_check";
CREATE TABLE "public"."tb_analysis_apply_check" (
  "check_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "model_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default",
  "param_data" "pg_catalog"."json",
  "param_auto_perfect" "pg_catalog"."varchar" COLLATE "pg_catalog"."default",
  "check_result" "pg_catalog"."text" COLLATE "pg_catalog"."default",
  "create_user" "pg_catalog"."varchar" COLLATE "pg_catalog"."default",
  "create_time" "pg_catalog"."timestamp"
)
;

-- ----------------------------
-- Table structure for tb_analysis_apply_file
-- ----------------------------
DROP TABLE IF EXISTS "public"."tb_analysis_apply_file";
CREATE TABLE "public"."tb_analysis_apply_file" (
  "file_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "file_name" "pg_catalog"."varchar" COLLATE "pg_catalog"."default",
  "file_path" "pg_catalog"."varchar" COLLATE "pg_catalog"."default",
  "demo" "pg_catalog"."text" COLLATE "pg_catalog"."default",
  "create_user" "pg_catalog"."varchar" COLLATE "pg_catalog"."default",
  "create_time" "pg_catalog"."timestamp"
)
;

-- ----------------------------
-- Table structure for tb_analysis_apply_sample
-- ----------------------------
DROP TABLE IF EXISTS "public"."tb_analysis_apply_sample";
CREATE TABLE "public"."tb_analysis_apply_sample" (
  "apply_sample_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "file_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default",
  "sample_name" "pg_catalog"."varchar" COLLATE "pg_catalog"."default",
  "sample_data" "pg_catalog"."text" COLLATE "pg_catalog"."default",
  "sample_state" "pg_catalog"."bpchar" COLLATE "pg_catalog"."default",
  "create_user" "pg_catalog"."varchar" COLLATE "pg_catalog"."default",
  "create_time" "pg_catalog"."timestamp"
)
;

-- ----------------------------
-- Table structure for tb_analysis_model
-- ----------------------------
DROP TABLE IF EXISTS "public"."tb_analysis_model";
CREATE TABLE "public"."tb_analysis_model" (
  "model_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "model_name" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "create_user" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "create_time" "pg_catalog"."timestamp",
  "param_config" "pg_catalog"."json"
)
;

-- ----------------------------
-- Table structure for tb_analysis_model_train
-- ----------------------------
DROP TABLE IF EXISTS "public"."tb_analysis_model_train";
CREATE TABLE "public"."tb_analysis_model_train" (
  "model_train_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "model_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "model_train_name" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "param_data" "pg_catalog"."json",
  "param_auto_perfect" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "model_train_data" "pg_catalog"."text" COLLATE "pg_catalog"."default",
  "create_user" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "create_time" "pg_catalog"."timestamp",
  "model_artifact_path" "pg_catalog"."varchar" COLLATE "pg_catalog"."default"
)
;

-- ----------------------------
-- Table structure for tb_analysis_model_train_process
-- ----------------------------
DROP TABLE IF EXISTS "public"."tb_analysis_model_train_process";
CREATE TABLE "public"."tb_analysis_model_train_process" (
  "model_train_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "happen_time" "pg_catalog"."timestamp",
  "process_data" "pg_catalog"."json",
  "create_user" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "create_time" "pg_catalog"."timestamp"
)
;

-- ----------------------------
-- Table structure for tb_analysis_model_train_sample
-- ----------------------------
DROP TABLE IF EXISTS "public"."tb_analysis_model_train_sample";
CREATE TABLE "public"."tb_analysis_model_train_sample" (
  "model_train_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "from_sample_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL DEFAULT NULL::character varying,
  "from_sample_type" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "create_user" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "create_time" "pg_catalog"."timestamp"
)
;

-- ----------------------------
-- Table structure for tb_analysis_sample_feature
-- ----------------------------
DROP TABLE IF EXISTS "public"."tb_analysis_sample_feature";
CREATE TABLE "public"."tb_analysis_sample_feature" (
  "feature_sample_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "from_sample_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "from_sample_type" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "feature_extract" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "feature_extract_param" "pg_catalog"."json",
  "feature_select" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "feature_sample_data" "pg_catalog"."text" COLLATE "pg_catalog"."default",
  "create_user" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "create_time" "pg_catalog"."timestamp"
)
;

-- ----------------------------
-- Table structure for tb_analysis_sample_file
-- ----------------------------
DROP TABLE IF EXISTS "public"."tb_analysis_sample_file";
CREATE TABLE "public"."tb_analysis_sample_file" (
  "file_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "file_name" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "file_path" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "demo" "pg_catalog"."text" COLLATE "pg_catalog"."default",
  "create_user" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "create_time" "pg_catalog"."timestamp" NOT NULL
)
;

-- ----------------------------
-- Table structure for tb_analysis_sample_original
-- ----------------------------
DROP TABLE IF EXISTS "public"."tb_analysis_sample_original";
CREATE TABLE "public"."tb_analysis_sample_original" (
  "original_sample_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "file_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "sample_name" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "sample_data" "pg_catalog"."text" COLLATE "pg_catalog"."default",
  "sample_state" "pg_catalog"."bpchar" COLLATE "pg_catalog"."default" DEFAULT '1'::bpchar,
  "create_user" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "create_time" "pg_catalog"."timestamp"
)
;

-- ----------------------------
-- Table structure for tb_analysis_sample_preprocess
-- ----------------------------
DROP TABLE IF EXISTS "public"."tb_analysis_sample_preprocess";
CREATE TABLE "public"."tb_analysis_sample_preprocess" (
  "preprocess_sample_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" NOT NULL,
  "original_sample_id" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "preprocess_method" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "preprocess_sample_data" "pg_catalog"."text" COLLATE "pg_catalog"."default",
  "create_user" "pg_catalog"."varchar" COLLATE "pg_catalog"."default" DEFAULT NULL::character varying,
  "create_time" "pg_catalog"."timestamp"
)
;

-- ----------------------------
-- Primary Key structure for table tb_analysis_apply_check
-- ----------------------------
ALTER TABLE "public"."tb_analysis_apply_check" ADD CONSTRAINT "tb_analysis_apply_check_pkey" PRIMARY KEY ("check_id");

-- ----------------------------
-- Primary Key structure for table tb_analysis_apply_file
-- ----------------------------
ALTER TABLE "public"."tb_analysis_apply_file" ADD CONSTRAINT "tb_analysis_apply_file_pkey" PRIMARY KEY ("file_id");

-- ----------------------------
-- Primary Key structure for table tb_analysis_apply_sample
-- ----------------------------
ALTER TABLE "public"."tb_analysis_apply_sample" ADD CONSTRAINT "tb_analysis_apply_sample_pkey" PRIMARY KEY ("apply_sample_id");

-- ----------------------------
-- Primary Key structure for table tb_analysis_model
-- ----------------------------
ALTER TABLE "public"."tb_analysis_model" ADD CONSTRAINT "tb_analysis_model_pkey" PRIMARY KEY ("model_id");

-- ----------------------------
-- Primary Key structure for table tb_analysis_model_train
-- ----------------------------
ALTER TABLE "public"."tb_analysis_model_train" ADD CONSTRAINT "tb_analysis_model_train_pkey" PRIMARY KEY ("model_train_id");

-- ----------------------------
-- Primary Key structure for table tb_analysis_model_train_process
-- ----------------------------
ALTER TABLE "public"."tb_analysis_model_train_process" ADD CONSTRAINT "tb_analysis_model_train_process_pkey" PRIMARY KEY ("model_train_id");

-- ----------------------------
-- Primary Key structure for table tb_analysis_model_train_sample
-- ----------------------------
ALTER TABLE "public"."tb_analysis_model_train_sample" ADD CONSTRAINT "tb_analysis_model_train_sample_pkey" PRIMARY KEY ("model_train_id", "from_sample_id");

-- ----------------------------
-- Primary Key structure for table tb_analysis_sample_feature
-- ----------------------------
ALTER TABLE "public"."tb_analysis_sample_feature" ADD CONSTRAINT "tb_analysis_sample_feature_pkey" PRIMARY KEY ("feature_sample_id");

-- ----------------------------
-- Primary Key structure for table tb_analysis_sample_file
-- ----------------------------
ALTER TABLE "public"."tb_analysis_sample_file" ADD CONSTRAINT "tb_analysis_sample_file_pkey" PRIMARY KEY ("file_id");

-- ----------------------------
-- Primary Key structure for table tb_analysis_sample_original
-- ----------------------------
ALTER TABLE "public"."tb_analysis_sample_original" ADD CONSTRAINT "tb_analysis_sample_original_pkey" PRIMARY KEY ("original_sample_id");

-- ----------------------------
-- Primary Key structure for table tb_analysis_sample_preprocess
-- ----------------------------
ALTER TABLE "public"."tb_analysis_sample_preprocess" ADD CONSTRAINT "tb_analysis_sample_preprocess_pkey" PRIMARY KEY ("preprocess_sample_id");

-- ----------------------------
-- Foreign Keys structure for table tb_analysis_apply_sample
-- ----------------------------
ALTER TABLE "public"."tb_analysis_apply_sample" ADD CONSTRAINT "tb_analysis_apply_sample_file_id_fkey" FOREIGN KEY ("file_id") REFERENCES "public"."tb_analysis_apply_file" ("file_id") ON DELETE NO ACTION ON UPDATE NO ACTION;

-- ----------------------------
-- Foreign Keys structure for table tb_analysis_model_train
-- ----------------------------
ALTER TABLE "public"."tb_analysis_model_train" ADD CONSTRAINT "fk_model_id" FOREIGN KEY ("model_id") REFERENCES "public"."tb_analysis_model" ("model_id") ON DELETE NO ACTION ON UPDATE NO ACTION;

-- ----------------------------
-- Foreign Keys structure for table tb_analysis_sample_original
-- ----------------------------
ALTER TABLE "public"."tb_analysis_sample_original" ADD CONSTRAINT "tb_analysis_sample_original_ibfk_1" FOREIGN KEY ("file_id") REFERENCES "public"."tb_analysis_sample_file" ("file_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
