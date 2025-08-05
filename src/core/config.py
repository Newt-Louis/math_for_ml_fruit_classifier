from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # lỗi kiểm tra nghiêm ngặt của pydantic-settings
    database_url: str
    api_key: str
    environment: str

    model_path: str = "./models/fruit_classifier_best.pt"
    image_size: tuple[int, int] = (224, 224)
    debug: bool = True

    class Config:
        env_file = ".env"

settings = Settings()
