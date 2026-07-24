import { createPrismError, ERROR_CODES } from './errors.js';

export const createRouteProjection = ({ routeForLocation, primaryRoute }) => ({
  project(route = routeForLocation(location.pathname)) {
    if (!route || !route.observation || !route.route) {
      throw createPrismError(
        ERROR_CODES.ROUTE_PROJECTION_FAILED,
        'Route projection failed: invalid route metadata',
        { route },
      );
    }
    document.body.dataset.observationRoute = route.observation;
    document.body.dataset.primaryRoute = String(primaryRoute(route) ?? 'reference');
    document.body.setAttribute('data-prism-observation-projected', 'true');
    document.body.dataset.prismObservationProjected = 'true';
    document.body.dataset.prismProjectionRoute = route.observation;
    return route;
  },
  projectForPath(pathname) {
    return routeForLocation(pathname);
  },
});
  }
});
