(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("dcda35f3-bab1-4e47-a1f1-dd881e977a92");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'dcda35f3-bab1-4e47-a1f1-dd881e977a92' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"87fae445-4fd4-439f-91f3-a0f593842d7d":{"roots":{"references":[{"attributes":{"below":[{"id":"21814"}],"center":[{"id":"21817"},{"id":"21821"}],"left":[{"id":"21818"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21880"},{"id":"21882"},{"id":"21883"},{"id":"21884"},{"id":"21887"}],"title":{"id":"21889"},"toolbar":{"id":"21832"},"toolbar_location":null,"x_range":{"id":"21806"},"x_scale":{"id":"21810"},"y_range":{"id":"21808"},"y_scale":{"id":"21812"}},"id":"21805","subtype":"Figure","type":"Plot"},{"attributes":{"callback":null},"id":"21829","type":"HoverTool"},{"attributes":{},"id":"21828","type":"SaveTool"},{"attributes":{},"id":"21825","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"21831"}},"id":"21826","type":"LassoSelectTool"},{"attributes":{},"id":"21918","type":"BasicTickFormatter"},{"attributes":{},"id":"21827","type":"UndoTool"},{"attributes":{},"id":"21906","type":"BasicTickFormatter"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21858"},{"id":"21859"},{"id":"21860"},{"id":"21861"},{"id":"21862"},{"id":"21863"},{"id":"21864"},{"id":"21865"}]},"id":"21868","type":"Toolbar"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21867","type":"PolyAnnotation"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21866","type":"BoxAnnotation"},{"attributes":{},"id":"21920","type":"BasicTickFormatter"},{"attributes":{"children":[[{"id":"21805"},0,0],[{"id":"21841"},0,1]]},"id":"21929","type":"GridBox"},{"attributes":{},"id":"21922","type":"Selection"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21830","type":"BoxAnnotation"},{"attributes":{},"id":"21908","type":"BasicTickFormatter"},{"attributes":{},"id":"21923","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21831","type":"PolyAnnotation"},{"attributes":{},"id":"21910","type":"Selection"},{"attributes":{},"id":"21911","type":"UnionRenderers"},{"attributes":{"data":{"rug_x":{"__ndarray__":"fV36E1z/6j89DycBWWfXP73VlTJ7YsE/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/Oqay45Jr6D86SHRMZcflP7HThSU1z+I/VdSaCTtd6D9Hvab2ZmSwP8UA5kQ6d4M/0GULqag1oz+aI/Yi4T7rP55DEK8H/NA/qgGd6qjX1D+VqnS/h2ThP/UnuP7VN+Q/TGXHJdeQ2z8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21924"},"selection_policy":{"id":"21925"}},"id":"21900","type":"ColumnDataSource"},{"attributes":{},"id":"21924","type":"Selection"},{"attributes":{},"id":"21925","type":"UnionRenderers"},{"attributes":{"data":{"rug_x":{"__ndarray__":"jQwCEA1Gsz9bBMLb9PjIP1Qd9Zram7E/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/aF85wG2piz9oXznAbamLP2hfOcBtqYs/aF85wG2piz9oXznAbamLP2hfOcBtqYs/pI3yRkqEyT9c/+ob+nG6P4OiBeyjALU/mx4fY+a33D9wF2c1cbTRPy51Aws2htg/P07TEgOYwz9FOncTGHy5P/BQ+ANPucc/uRO6PYJJzj9CRCWTDYpmP+UvVnGFsrI/8JR71fNwcj8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"21912"},"selection_policy":{"id":"21913"}},"id":"21886","type":"ColumnDataSource"},{"attributes":{},"id":"21912","type":"Selection"},{"attributes":{"source":{"id":"21886"}},"id":"21888","type":"CDSView"},{"attributes":{},"id":"21913","type":"UnionRenderers"},{"attributes":{"source":{"id":"21877"}},"id":"21881","type":"CDSView"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21885","type":"Dash"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2515582690238702},"id":"21882","type":"Span"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.1782444431478369},"id":"21883","type":"Span"},{"attributes":{"data_source":{"id":"21886"},"glyph":{"id":"21885"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21888"}},"id":"21887","type":"GlyphRenderer"},{"attributes":{"text":"tau"},"id":"21889","type":"Title"},{"attributes":{"source":{"id":"21891"}},"id":"21895","type":"CDSView"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21884","type":"Span"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21893","type":"Circle"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21892","type":"Circle"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"bcJe+Pxh3D88Eu5SUq3NPyL+eQAns8s/mzeOVbYJ1z+YvNPsmzTWP87j+UyletE/sIr1s8Bnzz8YN/msnr7PP8gPs0h4ec8/RNGUEkZ90z8g12riYrDUP+AlTxYjYc4/UBTOvdhAzD+AFwpyJ0DOPxDSzcUXbc4/kDNyJsikyj9AY0p3Si3PP8AWdsF70MQ/4I8dAXxLyD+grhW5nZrSPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21922"},"selection_policy":{"id":"21923"}},"id":"21891","type":"ColumnDataSource"},{"attributes":{"toolbars":[{"id":"21832"},{"id":"21868"}],"tools":[{"id":"21822"},{"id":"21823"},{"id":"21824"},{"id":"21825"},{"id":"21826"},{"id":"21827"},{"id":"21828"},{"id":"21829"},{"id":"21858"},{"id":"21859"},{"id":"21860"},{"id":"21861"},{"id":"21862"},{"id":"21863"},{"id":"21864"},{"id":"21865"}]},"id":"21930","type":"ProxyToolbar"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21822"},{"id":"21823"},{"id":"21824"},{"id":"21825"},{"id":"21826"},{"id":"21827"},{"id":"21828"},{"id":"21829"}]},"id":"21832","type":"Toolbar"},{"attributes":{"data_source":{"id":"21891"},"glyph":{"id":"21892"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21893"},"selection_glyph":null,"view":{"id":"21895"}},"id":"21894","type":"GlyphRenderer"},{"attributes":{"source":{"id":"21900"}},"id":"21902","type":"CDSView"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21899","type":"Dash"},{"attributes":{"data_source":{"id":"21877"},"glyph":{"id":"21878"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21879"},"selection_glyph":null,"view":{"id":"21881"}},"id":"21880","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2148430013731262},"id":"21896","type":"Span"},{"attributes":{"below":[{"id":"21850"}],"center":[{"id":"21853"},{"id":"21857"}],"left":[{"id":"21854"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21894"},{"id":"21896"},{"id":"21897"},{"id":"21898"},{"id":"21901"}],"title":{"id":"21903"},"toolbar":{"id":"21868"},"toolbar_location":null,"x_range":{"id":"21842"},"x_scale":{"id":"21846"},"y_range":{"id":"21844"},"y_scale":{"id":"21848"}},"id":"21841","subtype":"Figure","type":"Plot"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21918"},"ticker":{"id":"21851"}},"id":"21850","type":"LinearAxis"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.15209716424958658},"id":"21897","type":"Span"},{"attributes":{"end":1,"start":-0.05},"id":"21844","type":"DataRange1d"},{"attributes":{"data_source":{"id":"21900"},"glyph":{"id":"21899"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21902"}},"id":"21901","type":"GlyphRenderer"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21908"},"ticker":{"id":"21819"}},"id":"21818","type":"LinearAxis"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"21920"},"ticker":{"id":"21855"}},"id":"21854","type":"LinearAxis"},{"attributes":{"text":"mu"},"id":"21903","type":"Title"},{"attributes":{},"id":"21846","type":"LinearScale"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21878","type":"Circle"},{"attributes":{"children":[{"id":"21931"},{"id":"21929"}]},"id":"21932","type":"Column"},{"attributes":{},"id":"21848","type":"LinearScale"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"TOY7mm62yD/upazjteTNP/DqXfy8Lcs/HHCIRHVOzT9orzckCyTOP6y8CkfbP8w/iNx/9NF5yz+IfM+LFJ/NPygi8KloJ8w/oK0q3zNVzD/YcLH58jHPP6hWLqhoMNQ/IGaMl5nu0j8AebTCVJLRP+D0hUiIM80/uNpKBpu90z/A00Fq0J3TP2iCFjW8ldY/kLBgWUSo1j8QCPgZgPnXPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21910"},"selection_policy":{"id":"21911"}},"id":"21877","type":"ColumnDataSource"},{"attributes":{"callback":null},"id":"21865","type":"HoverTool"},{"attributes":{},"id":"21806","type":"DataRange1d"},{"attributes":{},"id":"21851","type":"BasicTicker"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21898","type":"Span"},{"attributes":{},"id":"21810","type":"LinearScale"},{"attributes":{"axis":{"id":"21850"},"ticker":null},"id":"21853","type":"Grid"},{"attributes":{},"id":"21822","type":"ResetTool"},{"attributes":{"end":1,"start":-0.05},"id":"21808","type":"DataRange1d"},{"attributes":{"toolbar":{"id":"21930"},"toolbar_location":"above"},"id":"21931","type":"ToolbarBox"},{"attributes":{"axis":{"id":"21854"},"dimension":1,"ticker":null},"id":"21857","type":"Grid"},{"attributes":{},"id":"21819","type":"BasicTicker"},{"attributes":{},"id":"21855","type":"BasicTicker"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21906"},"ticker":{"id":"21815"}},"id":"21814","type":"LinearAxis"},{"attributes":{},"id":"21812","type":"LinearScale"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21879","type":"Circle"},{"attributes":{"overlay":{"id":"21866"}},"id":"21860","type":"BoxZoomTool"},{"attributes":{},"id":"21815","type":"BasicTicker"},{"attributes":{},"id":"21859","type":"PanTool"},{"attributes":{},"id":"21858","type":"ResetTool"},{"attributes":{"axis":{"id":"21814"},"ticker":null},"id":"21817","type":"Grid"},{"attributes":{},"id":"21864","type":"SaveTool"},{"attributes":{},"id":"21861","type":"WheelZoomTool"},{"attributes":{"axis":{"id":"21818"},"dimension":1,"ticker":null},"id":"21821","type":"Grid"},{"attributes":{"overlay":{"id":"21867"}},"id":"21862","type":"LassoSelectTool"},{"attributes":{},"id":"21842","type":"DataRange1d"},{"attributes":{},"id":"21863","type":"UndoTool"},{"attributes":{"overlay":{"id":"21830"}},"id":"21824","type":"BoxZoomTool"},{"attributes":{},"id":"21823","type":"PanTool"}],"root_ids":["21932"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"87fae445-4fd4-439f-91f3-a0f593842d7d","root_ids":["21932"],"roots":{"21932":"dcda35f3-bab1-4e47-a1f1-dd881e977a92"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();