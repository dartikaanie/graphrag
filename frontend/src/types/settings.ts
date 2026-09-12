export interface SecretField {
  is_set: boolean
  masked: string | null
}

export interface SettingsState {
  provider: string
  model: string
  api_key: SecretField
  anthropic_api_key: SecretField
  ollama_host: string
  num_ctx: number
  neo4j_uri: string
  neo4j_user: string
  neo4j_password: SecretField
  neo4j_database: string
  questions_parquet: string
  answers_parquet: string
  judge_provider: string
  judge_model: string
  judge_temperature: number
  secondary_judge_provider: string
  secondary_judge_model: string
  kappa_sample_size: number
  judge_majority_rounds: number
}

export interface TestConnectionResult {
  ok: boolean
  message: string
}
