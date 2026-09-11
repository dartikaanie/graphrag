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
}

export interface TestConnectionResult {
  ok: boolean
  message: string
}
