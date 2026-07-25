import { createPrismError, ERROR_CODES } from './errors.js';

export const createRouteProjection = ({ routeForLocation, primaryRoute }) => ({
  project(route) {
    const projected = route?.route ? route : routeForLocation(String(route || '/'));
    if (!projected || !projected.observation || !projected.route) {
      throw createPrismError(
        ERROR_CODES.ROUTE_PROJECTION_FAILED,
        'Route projection failed: invalid route metadata',
        { route },
      );
    }
    const nextRoute = projected;
    document.body.dataset.observationRoute = nextRoute.observation;
    document.body.dataset.primaryRoute = String(primaryRoute(nextRoute) ?? 'reference');
    document.body.setAttribute('data-prism-observation-projected', 'true');
    document.body.dataset.prismObservationProjected = 'true';
    document.body.dataset.prismProjectionRoute = nextRoute.observation;
    return nextRoute;
  },
  projectForPath(pathname) {
    return routeForLocation(pathname);
  },
});
