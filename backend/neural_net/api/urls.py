from rest_framework.routers import DefaultRouter
from .views import LayerViewSet, NetworkViewSet


router = DefaultRouter()
router.register(r'layer', LayerViewSet, basename='neuralnetworklayers')
router.register(r'network', NetworkViewSet, basename='neuralnetwork')
urlpatterns = router.urls